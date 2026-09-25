"""Small CPU example of the function-space Laplace factory methods."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace


def main() -> None:
    torch.manual_seed(7)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = inputs.sum(dim=-1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    ella_model = torch.nn.Linear(2, 1)
    ella = Laplace(
        ella_model,
        "regression",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="nystrom",
        subsample_size=2,
        n_eigenvalues=1,
        sigma_noise=0.2,
    )
    ella.fit(loader)

    valla_model = torch.nn.Linear(2, 1)
    valla = Laplace(
        valla_model,
        "regression",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        inducing_locations="random",
        num_inducing=2,
        sigma_noise=0.2,
    )
    valla.fit(loader, iterations=2, lr=1e-3)

    for estimator in (ella, valla):
        mean, latent_covariance = estimator(inputs[:2], joint=True)
        print(type(estimator).__name__, mean.shape, latent_covariance.shape)


if __name__ == "__main__":
    main()
