# ELLA and VaLLA

ELLA and VaLLA approximate the function-space posterior of a pretrained network.
They are selected through the same `Laplace` factory as the subset-of-data GP
method. The default `functional_approximation="sod"` preserves the existing
`FunctionalLaplace` selection.

These methods follow [Deng, Zhou, and Zhu's ELLA](https://arxiv.org/abs/2210.12642)
and [Ortega, Rodríguez Santana, and Hernández-Lobato's VaLLA](https://arxiv.org/abs/2302.12565).
The implementations were adapted from BayesiPy's earlier ELLA and VaLLA ports,
then reworked to use Laplace's shared function-space API and curvature backends.

```python
from laplace import Laplace

ella = Laplace(
    model, "classification",
    subset_of_weights="all",
    hessian_structure="gp",
    functional_approximation="nystrom",
    subsample_size=100,
    n_eigenvalues=20,
)
ella.fit(train_loader)
probabilities = ella(x)  # [batch, classes]
logits, covariance = ella.predictive_moments(x)  # [batch, classes], [batch, classes, classes]
```

ELLA chooses a subset of training inputs, constructs a Nyström basis from
parameter Jacobians, and accumulates the generalized Gauss-Newton matrix in
that basis. Its `fit` returns `None`; processed example counts and validation
NLL values are in `fit_history_`. A validation grid search for scalar prior
precision is available through
`optimize_prior_precision(method="gridsearch", val_loader=...)`.

```python
valla = Laplace(
    model, "regression",
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

VaLLA optimizes its alpha-divergence data term plus KL over minibatches while
keeping the pretrained network fixed. Its prior precision and, for regression,
noise variance are trainable during `fit`. Inducing inputs can be supplied
directly or initialized with `"random"` or `"kmeans"`; k-means requires
floating-point tensor inputs. For classification and reward modeling,
`mc_softmax_samples > 0` estimates the alpha data term with latent samples;
the default uses the deterministic probit approximation for `alpha=1`.
Classification and reward modeling with another alpha require Monte Carlo
samples because the probit approximation does not depend on alpha.
Repeated `fit` calls reset VaLLA's inducing and variational state by default;
the learned prior precision and regression noise carry over.
Pass `override=False` to continue optimizing the fitted state and append to
`fit_history_`.

Both estimators accept `"classification"`, `"regression"`, and
`"reward_modeling"`. Reward modeling fits a pairwise classification
likelihood, then predicts the scalar reward and its covariance on individual
inputs. `estimator(pair_inputs, fitting=True)` returns pairwise class
probabilities. Their `functional_samples` and `predictive_samples` return
`[samples, batch, outputs]`; `joint=True` returns a full cross-input covariance
for regression and scalar reward prediction.
Sampling draws independent per-input marginals by default. Pass `joint=True`
to either sampling method to retain cross-input covariance.

The upstream `__call__` returns *latent* regression covariance without
observation noise. BayesiPy's `predict` adapter adds noise and reverses target
normalization. Neither ELLA nor VaLLA provides the subset-of-data GP marginal
likelihood, a tracked training log likelihood, or weight-space samples. Both provide `state_dict` and
`load_state_dict` for reuse with the same pretrained model and matching input
keys, curvature backend, backend options, and backpropagation setting. The
checkpoint restores the sampling seed (and VaLLA's random-generator state).

See the [ELLA](api_reference/ella.md) and
[VaLLA](api_reference/valla.md) API references.

## Reference behavior from the earlier ports

`tests/test_ella_valla.py` retains seeded values from BayesiPy commit `5ee24ed`.
For a fixed regression model and fixed inducing inputs, VaLLA's initial
objective, latent mean, and covariance agree with the earlier port. The old
`noise_variance=0.25` corresponds to Laplace's `sigma_noise=0.5` because the
latter is a standard deviation. ELLA preserves the pretrained model's logit
mean. Its earlier implementation omitted diagonal blocks of the Nyström basis
kernel, so its resulting covariance is intentionally not used as a reference.
