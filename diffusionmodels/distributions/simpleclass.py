"""
distributions.simpleclass
=========================

Provides various simple distributions

Classes
-------
MultivariateGaussian        : A multivariate normal distribution
InverseTransform            : A distribution defined by its CDF
"""


from ..utilities.warningsuppressors import unused_variables
from .functions import CumulativeDistributionFunction
from .inversion import InversionMethod
from .baseclass import Distribution
from typing import Optional
import torch


class MultivariateGaussian(Distribution):
    """
    A multivariate normal distribution

    Private Attributes
    ------------------
    `_mean: torch.Tensor`
        The mean vector of the multivariate Gaussian

    `_covariance: torch.Tensor`
        The covariance matrix of the multivariate Gaussian

    `_distribution: torch.distributions.MultivariateNormal`
        An instance of MultivariateNormal
    """

    def __init__(
        self,
        dimension: int,
        mean: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None
    ) -> None:
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

    def at(self, time: float) -> Distribution:
        unused_variables(time)
        return self

    def sample(self, num_samples: int) -> torch.Tensor:
        return self._distribution.sample((num_samples,))


class InverseTransform(Distribution):
    """
    A distribution defined by a CDF and sampled from using the inverse transform method

    Private Attributes
    ------------------
    `_cumulative_distribution_function: CumulativeDistributionFunction`
        The CDF that defines the data distribution

    `_inversion_method: InversionMethod`
        The method to find the inverse of CDFs

    `_device: torch.device`
        The device where all tensors are located
    """

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

    def at(self, time: float) -> Distribution:
        self._cumulative_distribution_function = self._cumulative_distribution_function.at(time)
        return self

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (num_samples, 1), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._cumulative_distribution_function,
            search_range = self._cumulative_distribution_function.boundaries()
        )
