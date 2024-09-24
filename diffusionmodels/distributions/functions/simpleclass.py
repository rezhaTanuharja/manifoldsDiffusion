"""
distributions.functions.simpleclass
===================================

Implements various simple PDFs and CDFs

Classes
-------
Linear
    The CDF of a uniform distribution (useless, just to test the interface)
"""


from ...utilities.warningsuppressors import unused_variables
from .baseclass import CumulativeDistributionFunction

from typing import Dict

import torch


class Linear(CumulativeDistributionFunction):
    """
    The CDF for uniform distributions (mainly just for interface testing)

    Private Attributes
    ------------------
    `_lower_bound: float`
        The highest value at which the CDF value is zero

    `_upper_bound: float`
        The lowest value at which the CDF value is one
    """

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self._upper_bound - self._lower_bound) * (
            -self._lower_bound + points
        )

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: float) -> CumulativeDistributionFunction:
        unused_variables(time)
        return self

    def boundaries(self) -> Dict[str, float]:
        return {
            'lower_bound': self._lower_bound,
            'upper_bound': self._upper_bound,
        }


class StepFunction(CumulativeDistributionFunction):

    def __init__(self, num_waves: int) -> None:
        self._num_waves = num_waves
        self._lower_bound = 0.0
        self._upper_bound = torch.pi

    def __call__(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[:, *[None for _ in range(points.dim())]]

        angles = wave_numbers * points[None, ...]

        return 1.0 / torch.pi * points + 2.0 / torch.pi * torch.sum(
            torch.sin(angles) / wave_numbers,
            dim = 0
        )

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: float) -> CumulativeDistributionFunction:
        unused_variables(time)
        return self

    def boundaries(self) -> Dict[str, float]:
        return {
            'lower_bound': self._lower_bound,
            'upper_bound': self._upper_bound,
        }





