"""Nyström approximation to the function-space Laplace posterior."""

from __future__ import annotations

from collections import deque
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader

from laplace._functional_utils import (
    check_model_fingerprint,
    classification_targets,
    model_fingerprint,
    preserve_model_gradients,
    regression_targets,
    split_batch,
    subset_loader,
    to_device,
    training_indices,
)
from laplace.baselaplace import BaseFunctionalLaplace, BaseLaplace
from laplace.curvature.curvature import CurvatureInterface
from laplace.utils.enums import (
    Likelihood,
    LinkApprox,
    PredType,
    PriorStructure,
    TuningMethod,
)


class ELLA(BaseFunctionalLaplace):
    """Accelerated linearized Laplace using a rank-limited Nyström feature map.

    `subsample_size` selects training inputs for the basis, while
    `n_eigenvalues` is its retained rank. `fit` accumulates projected GGN
    curvature and returns `None`. `fit_history_` records processed counts,
    validation NLL, and optional prior-grid scores. The pretrained model is
    evaluated without updating its parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Likelihood | str,
        subsample_size: int,
        n_eigenvalues: int,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        seed: int = 1234,
    ) -> None:
        if subsample_size <= 0 or n_eigenvalues <= 0 or n_eigenvalues > subsample_size:
            raise ValueError("Require 0 < n_eigenvalues <= subsample_size.")
        if torch.as_tensor(prior_precision).numel() != 1:
            raise ValueError("ELLA requires scalar prior_precision.")
        if torch.any(torch.as_tensor(prior_mean) != 0):
            raise ValueError("ELLA currently requires prior_mean=0.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        super().__init__(
            model,
            likelihood,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            temperature=temperature,
            enable_backprop=enable_backprop,
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        self.subsample_size = subsample_size
        self.n_eigenvalues = n_eigenvalues
        self.seed = seed
        self.dual_directions: torch.Tensor | None = None
        self.feature_gram: torch.Tensor | None = None
        self.chol_precision: torch.Tensor | None = None
        self._posterior_dirty = False
        self._fitted = False
        self.fit_history_: dict[str, list[Any]] = {
            "processed_examples": [],
            "val_nll": [],
            "tuning": [],
        }

    @property
    def prior_precision(self) -> torch.Tensor:
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, value: float | torch.Tensor) -> None:
        if torch.as_tensor(value).numel() != 1 or torch.any(
            torch.as_tensor(value) <= 0
        ):
            raise ValueError("ELLA requires positive scalar prior_precision.")
        BaseLaplace.prior_precision.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    @property
    def prior_mean(self) -> torch.Tensor:
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, value: float | torch.Tensor) -> None:
        if torch.any(torch.as_tensor(value) != 0):
            raise ValueError("ELLA currently requires prior_mean=0.")
        BaseLaplace.prior_mean.fset(self, value)  # type: ignore[attr-defined]

    @property
    def sigma_noise(self) -> torch.Tensor:
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, value: float | torch.Tensor) -> None:
        if torch.any(torch.as_tensor(value) <= 0):
            raise ValueError("sigma_noise must be positive.")
        if self.likelihood != Likelihood.REGRESSION and torch.any(
            torch.as_tensor(value) != 1
        ):
            raise ValueError("sigma_noise != 1 is only available for regression.")
        BaseLaplace.sigma_noise.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive.")
        self._temperature = value
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    def _batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        x, y = split_batch(batch, self.dict_key_y)
        return to_device(x, self._device, self._dtype), y.to(self._device)

    def _indices(self, loader: DataLoader, balanced: bool) -> torch.Tensor:
        available = training_indices(loader)
        n = len(available)
        if self.subsample_size > n:
            raise ValueError("subsample_size cannot exceed the training set size.")
        generator = torch.Generator().manual_seed(self.seed)
        if not balanced:
            chosen = torch.randperm(n, generator=generator)[: self.subsample_size]
            return available[chosen]
        if self.likelihood == Likelihood.REGRESSION:
            raise ValueError("balanced sampling is only defined for classification.")
        labels = []
        for index in available:
            _, target = split_batch(loader.dataset[int(index)], self.dict_key_y)
            labels.append(
                int(
                    torch.as_tensor(target).argmax()
                    if torch.as_tensor(target).numel() > 1
                    else target
                )
            )
        labels = torch.tensor(labels)
        classes = labels.unique(sorted=True)
        queues = []
        for cls in classes:
            candidates = torch.where(labels == cls)[0]
            order = torch.randperm(len(candidates), generator=generator)
            queues.append(deque(candidates[order].tolist()))
        selected: list[int] = []
        while len(selected) < self.subsample_size and any(queues):
            for queue in queues:
                if queue and len(selected) < self.subsample_size:
                    selected.append(queue.popleft())
        if len(selected) != self.subsample_size:
            raise ValueError("Not enough examples for balanced subsampling.")
        return available[torch.tensor(selected[: self.subsample_size])]

    def _build_basis(self, loader: DataLoader, indices: torch.Tensor) -> None:
        selected_rows = []
        generator = torch.Generator().manual_seed(self.seed + 1)
        for batch in subset_loader(loader, indices):
            x, _ = self._batch(batch)
            if (
                self.likelihood == Likelihood.REWARD_MODELING
                and "backpack" not in self._backend_cls.__name__.lower()
            ):
                jacobians, outputs = CurvatureInterface.jacobians(self.backend, x)
            else:
                jacobians, outputs = self.backend.jacobians(x)
            choices = torch.randint(
                outputs.shape[-1], (outputs.shape[0],), generator=generator
            ).to(jacobians.device)
            selected_rows.append(
                jacobians[torch.arange(len(choices), device=jacobians.device), choices]
            )
        rows = torch.cat(selected_rows)
        kernel = rows @ rows.T
        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        order = torch.argsort(eigenvalues, descending=True)[: self.n_eigenvalues]
        values = eigenvalues[order]
        if torch.any(values <= torch.finfo(values.dtype).eps):
            raise ValueError("Nyström subset kernel has insufficient positive rank.")
        self.dual_directions = rows.T @ (
            eigenvectors[:, order] / values.sqrt().unsqueeze(0)
        )

    def _features(
        self, x: torch.Tensor | MutableMapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dual_directions is None:
            raise RuntimeError("Call fit before requesting ELLA features.")
        # BackPACK extends modules in-place and is incompatible with torch.func.jvp.
        if "backpack" not in self._backend_cls.__name__.lower():
            parameters = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            feature_columns = []
            for direction in self.dual_directions.T:
                offset = 0
                tangents = {}
                for name, param in parameters.items():
                    if param.requires_grad:
                        tangent = direction[offset : offset + param.numel()].reshape_as(
                            param
                        )
                        offset += param.numel()
                    else:
                        tangent = torch.zeros_like(param)
                    tangents[name] = tangent

                def model_fn(params):
                    return torch.func.functional_call(
                        self.model, (params, buffers), (x,)
                    )

                try:
                    output, product = torch.func.jvp(
                        model_fn, (parameters,), (tangents,)
                    )
                except (NotImplementedError, RuntimeError) as exc:
                    if "forward AD" not in str(exc) and "jvp" not in str(exc):
                        raise
                    break
                feature_columns.append(product)
            if len(feature_columns) == self.n_eigenvalues:
                return torch.stack(feature_columns, dim=-1), output
        if (
            self.likelihood == Likelihood.REWARD_MODELING
            and "backpack" in self._backend_cls.__name__.lower()
            and (query_shape := self.model(x).shape)[-1] == 1
        ):
            selectors = torch.zeros(
                query_shape[0], device=self._device, dtype=torch.long
            )
            jacobians, output = self.backend.selected_output_jacobians(
                x, selectors, enable_backprop=self.enable_backprop
            )
        elif (
            self.likelihood == Likelihood.REWARD_MODELING
            and "backpack" not in self._backend_cls.__name__.lower()
        ):
            jacobians, output = CurvatureInterface.jacobians(
                self.backend, x, enable_backprop=self.enable_backprop
            )
        else:
            jacobians, output = self.backend.jacobians(
                x, enable_backprop=self.enable_backprop
            )
        return jacobians @ self.dual_directions, output

    def _refresh_posterior(self) -> None:
        if self._posterior_dirty:
            self._build_precision()

    def _build_precision(self) -> None:
        if self.feature_gram is None:
            return
        precision = self.feature_gram / self.temperature
        if self.likelihood == Likelihood.REGRESSION:
            precision = precision / self.sigma_noise.square()
        precision = precision + self.prior_precision * torch.eye(
            self.n_eigenvalues, device=self._device, dtype=self._dtype
        )
        self.chol_precision = torch.linalg.cholesky(precision)
        self._posterior_dirty = False

    @preserve_model_gradients
    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        val_steps: int | None = None,
        balanced: bool = False,
        progress_bar: bool = False,
    ) -> None:
        self.model.eval()
        self.fit_history_ = {"processed_examples": [], "val_nll": [], "tuning": []}
        self._fitted = False
        self.n_data = len(training_indices(train_loader))
        first_x, _ = self._batch(next(iter(train_loader)))
        if (
            isinstance(first_x, MutableMapping)
            and "backpack" in self._backend_cls.__name__.lower()
        ):
            raise ValueError(
                "BackPACK does not support mapping-style inputs; use AsdlGGN or CurvlinopsGGN."
            )
        self.n_outputs = self.model(first_x).shape[-1]
        setattr(self.model, "output_size", self.n_outputs)
        self._build_basis(train_loader, self._indices(train_loader, balanced))
        self.feature_gram = torch.zeros(
            self.n_eigenvalues,
            self.n_eigenvalues,
            device=self._device,
            dtype=self._dtype,
        )
        iterator = train_loader
        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(train_loader, desc="Fitting ELLA")
        processed = 0
        for step, batch in enumerate(iterator, start=1):
            x, y = self._batch(batch)
            features, output = self._features(x)
            if self.likelihood == Likelihood.REGRESSION:
                regression_targets(y, output)
                self.feature_gram += torch.einsum(
                    "bck,bcl->kl", features, features
                ).detach()
            else:
                probs = output.softmax(dim=-1)
                hessian = torch.diag_embed(probs) - probs.unsqueeze(
                    -1
                ) * probs.unsqueeze(-2)
                self.feature_gram += torch.einsum(
                    "bck,bcd,bdl->kl", features, hessian, features
                ).detach()
            processed += output.shape[0]
            self.fit_history_["processed_examples"].append(float(processed))
            if val_loader is not None and val_steps and step % val_steps == 0:
                self._build_precision()
                self._fitted = True
                self.fit_history_["val_nll"].append(
                    float(self._validation_nll(val_loader))
                )
        self._build_precision()
        self._fitted = True
        if val_loader is not None and (not val_steps or step % val_steps):
            self.fit_history_["val_nll"].append(float(self._validation_nll(val_loader)))

    def _feature_variance(self, features: torch.Tensor) -> torch.Tensor:
        solved = torch.cholesky_solve(features.transpose(1, 2), self.chol_precision)
        return features @ solved

    def _feature_covariance(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, self.n_eigenvalues)
        solved = torch.cholesky_solve(flat.T, self.chol_precision)
        return flat @ solved

    def functional_variance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return per-input covariance from raw parameter Jacobians."""
        self._check_fitted()
        self._check_jacobians(jacobians)
        return self._feature_variance(jacobians @ self.dual_directions)

    def functional_covariance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return joint covariance from raw parameter Jacobians."""
        self._check_fitted()
        self._check_jacobians(jacobians)
        return self._feature_covariance(jacobians @ self.dual_directions)

    @torch.enable_grad()
    def _glm_predictive_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = to_device(x, self._device, self._dtype)
        features, mean = self._features(x)
        covariance = (
            self._feature_covariance(features)
            if joint
            else self._feature_variance(features)
        )
        if joint:
            mean = mean.flatten()
        if not self.enable_backprop:
            return mean.detach(), covariance.detach()
        return mean, covariance

    def _validation_nll(self, loader: DataLoader) -> torch.Tensor:
        total = torch.zeros((), device=self._device, dtype=self._dtype)
        count = 0
        for batch in loader:
            x, y = self._batch(batch)
            if self.likelihood == Likelihood.REGRESSION:
                mean, covariance = self.predictive_moments(x)
                y = regression_targets(y, mean)
                variance = (
                    covariance.diagonal(dim1=-2, dim2=-1) + self.sigma_noise.square()
                )
                total += (
                    0.5
                    * (
                        (y - mean).square() / variance
                        + variance.log()
                        + torch.log(torch.as_tensor(2 * torch.pi, device=self._device))
                    )
                ).sum()
            else:
                probability = self(
                    x, fitting=self.likelihood == Likelihood.REWARD_MODELING
                )
                assert isinstance(probability, torch.Tensor)
                y = classification_targets(y)
                total += torch.nn.functional.nll_loss(
                    probability.log(), y, reduction="sum"
                )
            count += y.shape[0]
        return total / count

    def optimize_prior_precision(
        self,
        pred_type: PredType | str = PredType.GP,
        method: TuningMethod | str = TuningMethod.GRIDSEARCH,
        n_steps: int = 100,
        lr: float = 1e-1,
        init_prior_prec: float | torch.Tensor = 1.0,
        prior_structure: PriorStructure | str = PriorStructure.SCALAR,
        val_loader: DataLoader | None = None,
        loss: Any = None,
        log_prior_prec_min: float = -4,
        log_prior_prec_max: float = 4,
        grid_size: int = 100,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> None:
        self._check_fitted()
        if pred_type != PredType.GP:
            raise ValueError("ELLA supports only pred_type='gp'.")
        if (
            n_steps != 100
            or lr != 1e-1
            or torch.as_tensor(init_prior_prec).numel() != 1
            or torch.as_tensor(init_prior_prec).item() != 1.0
            or prior_structure != PriorStructure.SCALAR
            or loss is not None
            or link_approx != LinkApprox.PROBIT
            or n_samples != 100
            or verbose
            or progress_bar
        ):
            raise NotImplementedError(
                "ELLA supports validation grid search without marginal-likelihood options."
            )
        if method != TuningMethod.GRIDSEARCH:
            raise NotImplementedError(
                "ELLA supports validation grid search, not marginal likelihood."
            )
        if val_loader is None:
            raise ValueError("gridsearch requires val_loader.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        grid = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.optimize_hyperparameters(val_loader, grid)

    def optimize_hyperparameters(self, val_loader: DataLoader, grid) -> None:
        self._check_fitted()
        best_score = float("inf")
        best = None
        for candidate in grid:
            if isinstance(candidate, (tuple, list)):
                self.prior_precision, self.sigma_noise = candidate
            else:
                self.prior_precision = candidate
            score = float(self._validation_nll(val_loader))
            self.fit_history_["tuning"].append(
                {
                    "prior_precision": float(self.prior_precision),
                    "sigma_noise": float(self.sigma_noise),
                    "val_nll": score,
                }
            )
            if score < best_score:
                best_score, best = score, candidate
        if best is None:
            raise ValueError("Hyperparameter grid is empty.")
        if isinstance(best, (tuple, list)):
            self.prior_precision, self.sigma_noise = best
        else:
            self.prior_precision = best
        self._refresh_posterior()

    def log_marginal_likelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "ELLA does not use FunctionalLaplace's subset-of-data marginal likelihood."
        )

    @property
    def log_likelihood(self) -> torch.Tensor:
        raise NotImplementedError("ELLA does not accumulate a training log likelihood.")

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "cls_name": type(self).__name__,
            "n_params": self.n_params,
            "model_fingerprint": model_fingerprint(self.model),
            "likelihood": self.likelihood,
            "subsample_size": self.subsample_size,
            "n_eigenvalues": self.n_eigenvalues,
            "dual_directions": self.dual_directions.detach().clone()
            if self.dual_directions is not None
            else None,
            "feature_gram": self.feature_gram.detach().clone()
            if self.feature_gram is not None
            else None,
            "prior_precision": self.prior_precision,
            "sigma_noise": self.sigma_noise,
            "temperature": self.temperature,
            "n_data": self.n_data,
            "n_outputs": self.n_outputs,
            "fitted": self._fitted,
            "fit_history": deepcopy(self.fit_history_),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict.get("fitted", False):
            raise ValueError("ELLA checkpoint must contain a fitted posterior.")
        if (
            state_dict["cls_name"] != type(self).__name__
            or state_dict["n_params"] != self.n_params
        ):
            raise ValueError(
                "Checkpoint requires the same ELLA type and pretrained model."
            )
        if state_dict["likelihood"] != self.likelihood:
            raise ValueError("Checkpoint likelihood does not match.")
        check_model_fingerprint(self.model, state_dict["model_fingerprint"])
        self.subsample_size = state_dict["subsample_size"]
        self.n_eigenvalues = state_dict["n_eigenvalues"]
        self.dual_directions = state_dict["dual_directions"].to(self._device)
        self.feature_gram = state_dict["feature_gram"].to(self._device)
        self.prior_precision = state_dict["prior_precision"]
        self.sigma_noise = state_dict["sigma_noise"]
        self.temperature = state_dict["temperature"]
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        setattr(self.model, "output_size", self.n_outputs)
        self._fitted = state_dict["fitted"]
        self.fit_history_ = deepcopy(state_dict["fit_history"])
        self._build_precision()
        self.model.eval()
