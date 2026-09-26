# ELLA

ELLA approximates a pretrained network's function-space posterior with a
Nyström basis. This implementation follows [Deng, Zhou, and Zhu](https://arxiv.org/abs/2210.12642)
and adapts the earlier BayesiPy ELLA port to Laplace's curvature backends.

```python
from laplace import Laplace

ella = Laplace(
    model,
    "classification",
    subset_of_weights="all",
    hessian_structure="gp",
    functional_approximation="nystrom",
    subsample_size=100,
    n_eigenvalues=20,
)
ella.fit(train_loader)
probabilities = ella(x)  # [batch, classes]
logits, covariance = ella.predictive_moments(x)
```

`fit` selects training inputs, builds a Nyström basis from parameter Jacobians,
and accumulates the generalized Gauss–Newton matrix in that basis. It returns
`None`; processed example counts and validation NLL are available in
`fit_history_`. A validation grid search for scalar prior precision is available
through `optimize_prior_precision(method="gridsearch", val_loader=...)`.

ELLA supports classification, regression, and reward modeling. Reward modeling
fits pairwise preferences and predicts a scalar reward on individual inputs.
`ella(pair_inputs, fitting=True)` returns pairwise class probabilities. For
classification, `ella(x)` returns probabilities. For regression, it returns the
latent mean and covariance without observation noise. BayesiPy's `predict`
adapter adds observation noise and reverses target normalization.

`predictive_moments(x, joint=True)` returns cross-input covariance.
`functional_samples` and `predictive_samples` return tensors shaped
`[samples, batch, outputs]`; pass `joint=True` to preserve cross-input
dependence. Weight-space `sample`, the subset-of-data GP marginal likelihood,
and a tracked training log likelihood are unavailable.

`state_dict` and `load_state_dict` restore the fitted basis with an equivalent
pretrained model and matching input keys, curvature backend, backend options,
and backpropagation setting. The checkpoint also restores the sampling seed.

The seeded reference in `tests/test_ella.py` preserves the earlier BayesiPy
port's logit mean. That port omitted diagonal blocks of the Nyström basis
kernel, so its covariance is intentionally not used as a reference. Run
`pytest tests/test_ella.py` for CPU coverage and
`pytest -m cuda tests/test_ella_cuda.py` with CUDA PyTorch.

See the [ELLA API reference](api_reference/ella.md).
The standalone runnable example is `examples/ella.py` in the repository.
