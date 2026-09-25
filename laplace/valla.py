"""Variational sparse-GP approximation to linearized Laplace."""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any

import torch
from scipy.cluster.vq import kmeans2
from torch.utils.data import DataLoader

from laplace._functional_utils import (
    check_model_fingerprint,
    classification_targets,
    individual_reward_inputs,
    model_fingerprint,
    preserve_model_gradients,
    regression_targets,
    select_rows,
    split_batch,
    subset_loader,
    to_device,
    training_indices,
)
from laplace.baselaplace import BaseFunctionalLaplace, BaseLaplace
from laplace.curvature.curvature import CurvatureInterface
from laplace.utils.enums import Likelihood


class VaLLA(BaseFunctionalLaplace):
    """Variational linearized Laplace with inducing outputs and a fixed MAP model.

    `inducing_locations` accepts fixed tensor or mapping inputs, `"random"`,
    or `"kmeans"`. A strategy requires `num_inducing`. `alpha` is the
    alpha-divergence parameter in `[0, 1]`; zero uses the ELBO limit.
    `mc_softmax_samples` enables Monte Carlo classification data terms when
    positive. `fit` optimizes variational, inducing, prior, and regression
    noise parameters, returns `None`, and records objective and validation
    NLL values in `fit_history_`. Repeated fits reset variational and inducing
    state by default; `fit(..., override=False)` continues optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Likelihood | str,
        inducing_locations: torch.Tensor | MutableMapping | str,
        num_inducing: int | None = None,
        inducing_classes: torch.Tensor | None = None,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        mc_softmax_samples: int = 0,
    ) -> None:
        if torch.as_tensor(prior_precision).numel() != 1:
            raise ValueError("VaLLA requires scalar prior_precision.")
        if torch.any(torch.as_tensor(prior_mean) != 0):
            raise ValueError("VaLLA currently requires prior_mean=0.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not isinstance(mc_softmax_samples, int) or mc_softmax_samples < 0:
            raise ValueError("mc_softmax_samples must be a nonnegative integer.")
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
        if (
            self.likelihood != Likelihood.REGRESSION
            and alpha != 1
            and mc_softmax_samples == 0
        ):
            raise ValueError(
                "Classification alpha != 1 requires mc_softmax_samples > 0; "
                "the deterministic probit approximation is alpha-independent."
            )
        self.seed = seed
        self.alpha = alpha
        self.mc_softmax_samples = mc_softmax_samples
        self.generator = torch.Generator(device=self._device).manual_seed(seed)
        self.log_prior_precision = torch.nn.Parameter(
            self._prior_precision.detach().log().clone()
        )
        self.log_noise_variance = (
            torch.nn.Parameter(2 * self._sigma_noise.detach().log().clone())
            if likelihood == Likelihood.REGRESSION
            else None
        )
        self._inducing_strategy = (
            inducing_locations if isinstance(inducing_locations, str) else None
        )
        if self._inducing_strategy is not None and self._inducing_strategy not in {
            "random",
            "kmeans",
        }:
            raise ValueError(
                "inducing_locations must be inputs, 'random', or 'kmeans'."
            )
        if isinstance(inducing_locations, str):
            if num_inducing is None or num_inducing <= 0:
                raise ValueError(
                    "num_inducing must be positive for an inducing strategy."
                )
            self.num_inducing = num_inducing
            self.inducing_locations = None
        else:
            self.inducing_locations = self._make_inducing(inducing_locations)
            count = (
                self.inducing_locations[self.dict_key_x].shape[0]
                if isinstance(self.inducing_locations, MutableMapping)
                else self.inducing_locations.shape[0]
            )
            if num_inducing is not None and num_inducing != count:
                raise ValueError("num_inducing does not match inducing_locations.")
            self.num_inducing = count
        if self.num_inducing <= 0:
            raise ValueError("At least one inducing location is required.")
        self.inducing_classes = (
            torch.zeros(self.num_inducing, device=self._device, dtype=torch.long)
            if inducing_classes is None
            else torch.as_tensor(
                inducing_classes, device=self._device, dtype=torch.long
            )
        )
        if self.inducing_classes.shape != (self.num_inducing,):
            raise ValueError("inducing_classes must have one entry per inducing input.")
        self._initial_inducing_locations = (
            None
            if self._inducing_strategy is not None
            else self._clone_fixed_inputs(self.inducing_locations)
        )
        self._initial_inducing_classes = self.inducing_classes.detach().clone()
        self._init_variational_factor()
        self.n_data = 0
        self._fitted = False
        self.fit_history_: dict[str, list[float]] = {"objective": [], "val_nll": []}

    @property
    def prior_precision(self) -> torch.Tensor:
        if hasattr(self, "log_prior_precision"):
            return self.log_prior_precision.exp()
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, value: float | torch.Tensor) -> None:
        if torch.as_tensor(value).numel() != 1 or torch.any(
            torch.as_tensor(value) <= 0
        ):
            raise ValueError("VaLLA requires positive scalar prior_precision.")
        BaseLaplace.prior_precision.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "log_prior_precision"):
            with torch.no_grad():
                self.log_prior_precision.copy_(self._prior_precision.log())

    @property
    def prior_mean(self) -> torch.Tensor:
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, value: float | torch.Tensor) -> None:
        if torch.any(torch.as_tensor(value) != 0):
            raise ValueError("VaLLA currently requires prior_mean=0.")
        BaseLaplace.prior_mean.fset(self, value)  # type: ignore[attr-defined]

    @property
    def sigma_noise(self) -> torch.Tensor:
        log_noise_variance = getattr(self, "log_noise_variance", None)
        if isinstance(log_noise_variance, torch.Tensor):
            return (0.5 * log_noise_variance).exp()
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
        log_noise_variance = getattr(self, "log_noise_variance", None)
        if isinstance(log_noise_variance, torch.Tensor):
            with torch.no_grad():
                log_noise_variance.copy_(2 * self._sigma_noise.log())

    @staticmethod
    def _clone_fixed_inputs(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, MutableMapping):
            return {key: VaLLA._clone_fixed_inputs(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(VaLLA._clone_fixed_inputs(item) for item in value)
        return deepcopy(value)

    def _make_inducing(self, inputs: Any) -> torch.Tensor | dict:
        converted = to_device(inputs, self._device, self._dtype)
        if isinstance(converted, MutableMapping):
            return self._clone_fixed_inputs(converted)
        if not isinstance(converted, torch.Tensor):
            converted = torch.as_tensor(
                converted, device=self._device, dtype=self._dtype
            )
        if converted.is_floating_point():
            return torch.nn.Parameter(converted.detach().clone())
        return converted.detach().clone()

    def _init_variational_factor(self) -> None:
        rows, cols = torch.tril_indices(self.num_inducing, self.num_inducing)
        initial = torch.eye(self.num_inducing, device=self._device, dtype=self._dtype)
        self.L = torch.nn.Parameter(initial[rows, cols].clone())

    def _factor_matrix(self) -> torch.Tensor:
        rows, cols = torch.tril_indices(
            self.num_inducing, self.num_inducing, device=self._device
        )
        matrix = torch.zeros(
            self.num_inducing, self.num_inducing, device=self._device, dtype=self._dtype
        )
        matrix[rows, cols] = self.L
        return matrix

    def _batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        x, y = split_batch(batch, self.dict_key_y)
        return to_device(x, self._device, self._dtype), y.to(self._device)

    def _gather_inducing(self, loader: DataLoader) -> None:
        candidates = []
        classes = []
        source = loader
        if self._inducing_strategy == "random":
            available_indices = training_indices(loader)
            n_pairs = (self.num_inducing + 1) // 2
            needed = (
                n_pairs
                if self.likelihood == Likelihood.REWARD_MODELING
                else self.num_inducing
            )
            count = min(len(available_indices), needed)
            choices = torch.randperm(
                len(available_indices),
                generator=torch.Generator().manual_seed(self.seed),
            )[:count]
            indices = available_indices[choices]
            source = subset_loader(loader, indices)
        for batch in source:
            x, y = self._batch(batch)
            if self.likelihood == Likelihood.REWARD_MODELING:
                x = individual_reward_inputs(x, self.dict_key_x)
                labels = torch.zeros(
                    x[self.dict_key_x].shape[0]
                    if isinstance(x, MutableMapping)
                    else x.shape[0],
                    device=self._device,
                    dtype=torch.long,
                )
            elif self.likelihood == Likelihood.CLASSIFICATION:
                labels = classification_targets(y)
            else:
                labels = torch.zeros(y.shape[0], device=self._device, dtype=torch.long)
            candidates.append(x)
            classes.append(labels)
        if not candidates:
            raise ValueError("Cannot initialize inducing inputs from an empty loader.")
        if isinstance(candidates[0], MutableMapping):
            inputs = {
                key: torch.cat([candidate[key] for candidate in candidates], dim=0)
                for key in candidates[0]
            }
        else:
            inputs = torch.cat(candidates, dim=0)
        labels = torch.cat(classes)
        available = labels.shape[0]
        if available < self.num_inducing:
            raise ValueError("num_inducing exceeds available training inputs.")
        if self._inducing_strategy == "kmeans":
            if isinstance(inputs, MutableMapping) or not inputs.is_floating_point():
                raise ValueError("kmeans requires floating-point tensor inputs.")
            flat = inputs.detach().cpu().reshape(available, -1).numpy()
            centers, _ = kmeans2(
                flat, self.num_inducing, minit="points", seed=self.seed
            )
            distances = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            indices = torch.as_tensor(distances.argmin(0), device=self._device)
        else:
            indices = torch.randperm(
                available, generator=torch.Generator().manual_seed(self.seed)
            )[: self.num_inducing].to(self._device)
        self.inducing_locations = self._make_inducing(select_rows(inputs, indices))
        self.inducing_classes = labels[indices].long()

    def _query_jacobians(
        self, x: torch.Tensor | MutableMapping, *, differentiable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "asdl" in self._backend_cls.__name__.lower() or (
            self.likelihood == Likelihood.REWARD_MODELING
            and "backpack" not in self._backend_cls.__name__.lower()
        ):
            return CurvatureInterface.jacobians(
                self.backend, x, enable_backprop=differentiable
            )
        output_size = self.model(x).shape[-1]
        previous = getattr(self.model, "output_size", None)
        setattr(self.model, "output_size", output_size)
        try:
            return self.backend.jacobians(x, enable_backprop=differentiable)
        finally:
            if previous is None:
                delattr(self.model, "output_size")
            else:
                setattr(self.model, "output_size", previous)

    def _inducing_jacobians(self) -> torch.Tensor:
        if self.inducing_locations is None:
            raise RuntimeError("Inducing locations have not been initialized.")
        selected, _ = self.backend.selected_output_jacobians(
            self.inducing_locations,
            self.inducing_classes,
            enable_backprop=isinstance(self.inducing_locations, torch.nn.Parameter),
        )
        return selected[:, 0, :]

    def _inducing_term(
        self, jacobians: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernel = jacobians @ jacobians.T / self.prior_precision
        factor = self._factor_matrix()
        identity = torch.eye(self.num_inducing, device=self._device, dtype=self._dtype)
        hessian = identity + factor.T @ kernel @ factor
        correction = factor @ torch.linalg.solve(hessian, factor.T)
        return kernel, hessian, correction

    def _latent_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = to_device(x, self._device, self._dtype)
        jacobians, mean = self._query_jacobians(x, differentiable=self.enable_backprop)
        inducing_jacobians = self._inducing_jacobians()
        kernel_zz, hessian, correction = self._inducing_term(inducing_jacobians)
        if joint:
            flat = jacobians.reshape(-1, self.n_params)
            prior = flat @ flat.T / self.prior_precision
            cross = flat @ inducing_jacobians.T / self.prior_precision
            covariance = prior - cross @ correction @ cross.T
        else:
            prior = (
                torch.einsum("bcp,bdp->bcd", jacobians, jacobians)
                / self.prior_precision
            )
            cross = (
                torch.einsum("bcp,mp->bcm", jacobians, inducing_jacobians)
                / self.prior_precision
            )
            covariance = prior - torch.einsum(
                "bcm,mn,bdn->bcd", cross, correction, cross
            )
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        return mean, covariance, kernel_zz, hessian, correction

    @torch.enable_grad()
    def _glm_predictive_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, covariance, _, _, _ = self._latent_distribution(x, joint)
        if joint:
            mean = mean.flatten()
        if not self.enable_backprop:
            return mean.detach(), covariance.detach()
        return mean, covariance

    def functional_variance(self, jacobians: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        self._check_jacobians(jacobians)
        inducing = self._inducing_jacobians()
        _, _, correction = self._inducing_term(inducing)
        prior = (
            torch.einsum("bcp,bdp->bcd", jacobians, jacobians) / self.prior_precision
        )
        cross = jacobians @ inducing.T / self.prior_precision
        return prior - torch.einsum("bcm,mn,bdn->bcd", cross, correction, cross)

    def functional_covariance(self, jacobians: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        self._check_jacobians(jacobians)
        inducing = self._inducing_jacobians()
        _, _, correction = self._inducing_term(inducing)
        flat = jacobians.reshape(-1, self.n_params)
        prior = flat @ flat.T / self.prior_precision
        cross = flat @ inducing.T / self.prior_precision
        return prior - cross @ correction @ cross.T

    def _objective(self, x: Any, y: torch.Tensor) -> torch.Tensor:
        mean, covariance, kernel, hessian, correction = self._latent_distribution(x)
        if self.likelihood == Likelihood.REGRESSION:
            y = regression_targets(y, mean)
            noise_variance = self.sigma_noise.square()
            latent_variance = covariance.diagonal(dim1=-2, dim2=-1)
            squared_error = (y - mean).square()
            if self.alpha == 0:
                energy = (squared_error + latent_variance) / noise_variance
            else:
                energy = (
                    squared_error / (noise_variance + self.alpha * latent_variance)
                    + torch.log1p(self.alpha * latent_variance / noise_variance)
                    / self.alpha
                )
            log_term = -0.5 * (energy + torch.log(2 * torch.pi * noise_variance)).sum()
        else:
            y = classification_targets(y)
            if self.mc_softmax_samples:
                draws = self._draw_gaussian(
                    mean, covariance, self.mc_softmax_samples, self.generator
                )
                log_probabilities = draws.log_softmax(dim=-1)
                selected = log_probabilities.gather(
                    -1,
                    y.long().view(1, -1, 1).expand(self.mc_softmax_samples, -1, 1),
                ).squeeze(-1)
                if self.alpha == 0:
                    log_term = selected.mean(dim=0).sum()
                else:
                    log_term = (
                        torch.logsumexp(self.alpha * selected, dim=0)
                        - torch.log(
                            torch.as_tensor(
                                self.mc_softmax_samples,
                                device=self._device,
                                dtype=self._dtype,
                            )
                        )
                    ).sum() / self.alpha
            else:
                scaled = mean / torch.sqrt(
                    1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1)
                )
                log_term = -torch.nn.functional.cross_entropy(
                    scaled, y.long(), reduction="sum"
                )
        kl = 0.5 * (
            torch.linalg.slogdet(hessian).logabsdet - torch.trace(kernel @ correction)
        )
        return -(self.n_data / y.shape[0]) * log_term / self.temperature + kl

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

    @preserve_model_gradients
    def fit(
        self,
        train_loader: DataLoader,
        *,
        iterations: int = 100,
        lr: float = 1e-3,
        val_loader: DataLoader | None = None,
        val_steps: int | None = None,
        progress_bar: bool = False,
        override: bool = True,
    ) -> None:
        if iterations <= 0 or lr <= 0:
            raise ValueError("iterations and lr must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not isinstance(self.mc_softmax_samples, int) or self.mc_softmax_samples < 0:
            raise ValueError("mc_softmax_samples must be a nonnegative integer.")
        if (
            self.likelihood != Likelihood.REGRESSION
            and self.alpha != 1
            and self.mc_softmax_samples == 0
        ):
            raise ValueError(
                "Classification alpha != 1 requires mc_softmax_samples > 0."
            )
        self.model.eval()
        self.n_data = len(training_indices(train_loader))
        x_first, _ = self._batch(next(iter(train_loader)))
        if (
            isinstance(x_first, MutableMapping)
            and "backpack" in self._backend_cls.__name__.lower()
        ):
            raise ValueError(
                "BackPACK does not support mapping-style inputs; use AsdlGGN or CurvlinopsGGN."
            )
        self.n_outputs = self.model(x_first).shape[-1]
        setattr(self.model, "output_size", self.n_outputs)
        if override or not self._fitted:
            if self._inducing_strategy is not None:
                self._gather_inducing(train_loader)
            else:
                self.inducing_locations = self._make_inducing(
                    self._initial_inducing_locations
                )
                self.inducing_classes = self._initial_inducing_classes.detach().clone()
            self._init_variational_factor()
            self.fit_history_ = {"objective": [], "val_nll": []}
        if (
            self.inducing_classes.shape != (self.num_inducing,)
            or torch.any(self.inducing_classes < 0)
            or torch.any(self.inducing_classes >= self.n_outputs)
        ):
            raise ValueError(
                "inducing_classes must contain one valid output per inducing input."
            )
        self._fitted = False
        parameters = [self.L, self.log_prior_precision]
        if self.log_noise_variance is not None:
            parameters.append(self.log_noise_variance)
        if isinstance(self.inducing_locations, torch.nn.Parameter):
            parameters.append(self.inducing_locations)
        optimizer = torch.optim.Adam(parameters, lr=lr)
        iterator = range(iterations)
        if progress_bar:
            from tqdm import trange

            iterator = trange(iterations, desc="Fitting VaLLA")
        batches = iter(train_loader)
        for step in iterator:
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                batch = next(batches)
            x, y = self._batch(batch)
            optimizer.zero_grad()
            objective = self._objective(x, y)
            objective.backward()
            optimizer.step()
            self.fit_history_["objective"].append(float(objective.detach()))
            if val_loader is not None and val_steps and (step + 1) % val_steps == 0:
                self._fitted = True
                with torch.no_grad():
                    self.fit_history_["val_nll"].append(
                        float(self._validation_nll(val_loader))
                    )
        self._fitted = True
        if val_loader is not None and (not val_steps or iterations % val_steps):
            with torch.no_grad():
                self.fit_history_["val_nll"].append(
                    float(self._validation_nll(val_loader))
                )

    def optimize_prior_precision(self, *args, **kwargs) -> None:
        raise NotImplementedError("VaLLA optimizes prior precision during fit.")

    def log_marginal_likelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "VaLLA's variational objective is not a Laplace marginal likelihood."
        )

    @property
    def log_likelihood(self) -> torch.Tensor:
        raise NotImplementedError("VaLLA optimizes an alpha-divergence objective.")

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "cls_name": type(self).__name__,
            "n_params": self.n_params,
            "model_fingerprint": model_fingerprint(self.model),
            "likelihood": self.likelihood,
            "inducing_locations": self.inducing_locations.detach().clone()
            if isinstance(self.inducing_locations, torch.Tensor)
            else deepcopy(self.inducing_locations),
            "inducing_classes": self.inducing_classes.detach().clone(),
            "inducing_strategy": self._inducing_strategy,
            "initial_inducing_locations": self._clone_fixed_inputs(
                self._initial_inducing_locations
            ),
            "initial_inducing_classes": self._initial_inducing_classes.detach().clone(),
            "L": self.L.detach().clone(),
            "log_prior_precision": self.log_prior_precision.detach().clone(),
            "log_noise_variance": None
            if self.log_noise_variance is None
            else self.log_noise_variance.detach().clone(),
            "temperature": self.temperature,
            "alpha": self.alpha,
            "mc_softmax_samples": self.mc_softmax_samples,
            "n_data": self.n_data,
            "n_outputs": self.n_outputs,
            "fitted": self._fitted,
            "fit_history": deepcopy(self.fit_history_),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict.get("fitted", False):
            raise ValueError("VaLLA checkpoint must contain a fitted posterior.")
        if (
            state_dict["cls_name"] != type(self).__name__
            or state_dict["n_params"] != self.n_params
        ):
            raise ValueError(
                "Checkpoint requires the same VaLLA type and pretrained model."
            )
        if state_dict["likelihood"] != self.likelihood:
            raise ValueError("Checkpoint likelihood does not match.")
        check_model_fingerprint(self.model, state_dict["model_fingerprint"])
        self.inducing_locations = self._make_inducing(state_dict["inducing_locations"])
        self.inducing_classes = state_dict["inducing_classes"].to(self._device)
        self.num_inducing = len(self.inducing_classes)
        self._inducing_strategy = state_dict["inducing_strategy"]
        self._initial_inducing_locations = self._clone_fixed_inputs(
            to_device(
                state_dict["initial_inducing_locations"], self._device, self._dtype
            )
        )
        self._initial_inducing_classes = state_dict["initial_inducing_classes"].to(
            self._device
        )
        self._init_variational_factor()
        with torch.no_grad():
            self.L.copy_(state_dict["L"].to(self._device))
            self.log_prior_precision.copy_(
                state_dict["log_prior_precision"].to(self._device)
            )
            if self.log_noise_variance is not None:
                self.log_noise_variance.copy_(
                    state_dict["log_noise_variance"].to(self._device)
                )
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        self.temperature = state_dict["temperature"]
        self.alpha = state_dict["alpha"]
        self.mc_softmax_samples = state_dict["mc_softmax_samples"]
        setattr(self.model, "output_size", self.n_outputs)
        self._fitted = state_dict["fitted"]
        self.fit_history_ = deepcopy(state_dict["fit_history"])
        self.model.eval()
