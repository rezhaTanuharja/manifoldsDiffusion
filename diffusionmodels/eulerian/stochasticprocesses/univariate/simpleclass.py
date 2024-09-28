"""
eulerian.stochasticprocesses.univariate.simpleclass
===================================================

Implements various simple univariate stochastic processes

Classes
-------
InverseTransform    : A stochastic process that is defined by its CDF
"""


from ..interfaces import StochasticProcess
from .functions import CumulativeDistributionFunction
from .inversion import InversionMethod

import torch

from typing import Callable, Tuple


class InverseTransform(StochasticProcess):
    """
    A stochastic process that is defined by its CDF and sampled from using the inverse transform method

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
        distribution_function: CumulativeDistributionFunction,
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

    def at(self, time: torch.Tensor) -> StochasticProcess:
        self._distribution_function = self._distribution_function.at(time)
        self._num_times = time.numel()
        return self

    def density(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._distribution_function.gradient

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (self._num_times, num_samples), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._distribution_function,
            search_range = self._distribution_function.support()
        )
