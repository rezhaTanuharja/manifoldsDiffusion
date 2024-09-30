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

from typing import Tuple


class InverseTransform(StochasticProcess):
    """
    A stochastic process that is defined by its CDF and sampled from using the inverse transform method
    """

    def __init__(
        self,
        cumulative_distribution_function: CumulativeDistributionFunction,
        inversion_method: InversionMethod
    ) -> None:
        """
        Parameters
        ----------
        `cumulative_distribution_function: CumulativeDistributionFunction`
            The CDF that defines the data distribution

        `inversion_method: InversionMethod`
            The method to find the inverse of CDFs
        """
        self._cumulative_distribution_function = cumulative_distribution_function
        self._inversion_method = inversion_method
        self._num_times = 1
        self._device = torch.device('cpu')

    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def to(self, device: torch.device):
        self._device = device

        self._cumulative_distribution_function.to(device)
        self._inversion_method.to(device)

    def at(self, time: torch.Tensor) -> StochasticProcess:
        self._cumulative_distribution_function = self._cumulative_distribution_function.at(time)
        self._num_times = time.numel()
        return self

    def density(self, points: torch.Tensor) -> torch.Tensor:
        return self._cumulative_distribution_function.gradient(points)

    def score_function(self, points: torch.Tensor) -> torch.Tensor:
        return (
            self._cumulative_distribution_function.hessian(points)
            /
            self._cumulative_distribution_function.gradient(points)
        )

    def sample(self, num_samples: int) -> torch.Tensor:

        values = torch.rand(size = (self._num_times, num_samples), device = self._device)

        return self._inversion_method.solve(
            values = values,
            function = self._cumulative_distribution_function,
            search_range = self._cumulative_distribution_function.support()
        )
