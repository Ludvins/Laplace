# VaLLA

VaLLA fits a variational function-space approximation around a fixed pretrained
network. This implementation follows [Ortega, Rodríguez Santana, and
Hernández-Lobato](https://arxiv.org/abs/2302.12565) and adapts the earlier
BayesiPy VaLLA port to Laplace's curvature backends.

```python
from laplace import Laplace

valla = Laplace(
    model,
    "regression",
    subset_of_weights="all",
    hessian_structure="gp",
    functional_approximation="variational",
    inducing_locations="random",
    num_inducing=32,
    sigma_noise=0.1,  # observation standard deviation
    alpha=1.0,  # use 0.0 for the ELBO limit
)
valla.fit(train_loader, iterations=100, lr=1e-3)
mean, latent_covariance = valla(x)
objective_history = valla.fit_history_["objective"]
```

`fit` optimizes the alpha-divergence data term plus KL over minibatches while
keeping network weights fixed. It returns `None`; objective and validation
results are stored in `fit_history_`. Prior precision and, for regression,
noise variance can change during fitting. Supply inducing inputs directly or
initialize them with `"random"` or `"kmeans"`; k-means requires floating-point
tensor inputs. Repeated `fit` calls reset inducing and variational state by
default while carrying over learned prior precision and regression noise. Pass
`override=False` to continue and append to `fit_history_`.

VaLLA supports classification, regression, and reward modeling. For
classification and reward modeling, `mc_softmax_samples > 0` estimates the
alpha data term using latent samples. The default uses a deterministic probit
approximation for `alpha=1`; other alpha values require Monte Carlo samples.
Reward modeling fits pairwise preferences and predicts scalar rewards on
individual inputs. `valla(pair_inputs, fitting=True)` returns pairwise class
probabilities.

For classification, `valla(x)` returns probabilities. For regression, it
returns the latent mean and covariance without observation noise; BayesiPy's
`predict` adapter adds noise and reverses target normalization.
`predictive_moments(x, joint=True)` returns cross-input covariance.
`functional_samples` and `predictive_samples` return
`[samples, batch, outputs]`; pass `joint=True` to retain cross-input
dependence. Weight-space `sample`, the subset-of-data GP marginal likelihood,
and a tracked training log likelihood are unavailable.

`state_dict` and `load_state_dict` restore the variational and inducing state
with an equivalent pretrained model and matching input keys, curvature
backend, backend options, and backpropagation setting. The checkpoint also
restores the sampling seed and random-generator state.

The seeded reference in `tests/test_valla.py` checks the earlier BayesiPy port's
initial regression objective, latent mean, and covariance. Its old
`noise_variance=0.25` corresponds to Laplace's `sigma_noise=0.5` because the
latter is a standard deviation. Run `pytest tests/test_valla.py` for CPU
coverage and `pytest -m cuda tests/test_valla_cuda.py` with CUDA PyTorch.

See the [VaLLA API reference](api_reference/valla.md).
The standalone runnable example is `examples/valla.py` in the repository.
