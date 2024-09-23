"""
distributions.simpleclass
=========================

Provides various simple distributions

Classes
-------
MultivariateGaussian        : A multivariate normal distribution
InverseTransform            : A distribution defined by its CDF and sampled using the inverse transform method
"""


from diffusionmodels.distributions.functions.baseclass import CumulativeDistributionFunction
from diffusionmodels.distributions.inversion.baseclass import InversionMethod
from .baseclass import Distribution
from typing import Optional
import torch


class MultivariateGaussian(Distribution):

    def __init__(
        self,
        dimension: int,
        mean: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None
    ) -> None:
        self._dimension = dimension

        if mean == None or mean.numel() != dimension:
            mean = torch.zeros(size = (dimension,))

        self._mean = mean

        if covariance == None or covariance.shape != (dimension, dimension):
            covariance = torch.eye(dimension)

        self._covariance = covariance

        self._distribution = torch.distributions.MultivariateNormal(mean, covariance)

    def to(self, device: torch.device):

        self._mean = self._mean.to(device)
        self._covariance = self._covariance.to(device)

        self._distribution = torch.distributions.MultivariateNormal(self._mean, self._covariance)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self._distribution.sample((num_samples,))


class InverseTransform(Distribution):

    def __init__(
        self,
        cumulative_distribution_function: CumulativeDistributionFunction,
        inversion_method: InversionMethod
    ) -> None:
        self._cumulative_distribution_function = cumulative_distribution_function
        self._inversion_method = inversion_method
        self._device = torch.device('cpu')

    def to(self, device: torch.device):
        self._device = device

        self._cumulative_distribution_function.to(device)
        self._inversion_method.to(device)

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (num_samples, 1), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._cumulative_distribution_function
        )
