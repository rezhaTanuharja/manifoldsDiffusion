"""
diffusionmodels.distributions.univariate.simpleclass
====================================================

Implements various simple univariate distributions

Classes
-------
InverseTransform    : Distribution that can be sampled by inverting their CDFs
"""


from ..interfaces import Distribution
from .functions import DistributionFunction
from .inversion import InversionMethod

import torch

from typing import Callable, Tuple


class InverseTransform(Distribution):
    """
    A distribution defined by a CDF and sampled from using the inverse transform method

    Private Attributes
    ------------------
    `_cumulative_distribution_function: CumulativeDistributionFunction`
        The CDF that defines the data distribution

    `_inversion_method: InversionMethod`
        The method to find the inverse of CDFs

    `_num_times: int`
        The number of time the distribution will be evaluated at

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
        self._num_times = 1
        self._device = torch.device('cpu')

    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def to(self, device: torch.device):
        self._device = device

        self._distribution_function.to(device)
        self._inversion_method.to(device)

    def at(self, time: torch.Tensor) -> Distribution:
        self._distribution_function = self._distribution_function.at(time)
        self._num_times = time.numel()
        return self

    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._distribution_function.density

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (self._num_times, num_samples), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._distribution_function.cumulative,
            search_range = self._distribution_function.support()
        )
