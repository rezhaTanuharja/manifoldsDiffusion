"""
distributions.simpleclass
=========================

Provides various simple distributions

Classes
-------
InverseTransform            : A distribution defined by its distribution function
UniformSphere               : A uniform distribution on unit spheres
"""


from ..utilities.warningsuppressors import unused_variables
from .functions import DistributionFunction
from .inversion import InversionMethod
from .baseclass import Distribution
from typing import Callable
import torch


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
        distribution_function: DistributionFunction,
        inversion_method: InversionMethod
    ) -> None:
        self._distribution_function = distribution_function
        self._inversion_method = inversion_method
        self._device = torch.device('cpu')

    def to(self, device: torch.device):
        self._device = device

        self._distribution_function.to(device)
        self._inversion_method.to(device)

    def at(self, time: float) -> Distribution:
        self._distribution_function = self._distribution_function.at(time)
        return self

    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._distribution_function.density

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (num_samples, 1), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._distribution_function.cumulative,
            search_range = self._distribution_function.support()
        )


class UniformSphere(Distribution):
    """
    A uniform distribution of points on an arbitrary-dimension unit sphere

    Private Attributes
    ------------------
    `_dimension: int`
        The dimension of the unit sphere

    `_device: torch.device`
        The device where all tensors are stored or created
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._device = torch.device('cpu')

    def to(self, device: torch.device):
        self._device = device

    def at(self, time: float) -> Distribution:
        unused_variables(time)
        return self

    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda _: torch.exp(

            torch.special.gammaln(0.5 * self._dimension)

        ) / (2.0 * torch.pi ** (0.5 * self._dimension))

    def sample(self, num_samples: int) -> torch.Tensor:

        points = torch.randn(
            size = (num_samples, self._dimension),
            device = self._device
        )

        return points / (torch.norm(points, dim = -1, keepdim = True) + 1e-6)
