"""
distributions.functions.simpleclass
===================================

Implements various simple PDFs and CDFs

Classes
-------
Linear
    The CDF of a uniform distribution (useless, just to test the interface)

Heaviside
    The CDF associated with a dirac distribution at the origin
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


class Heaviside(CumulativeDistributionFunction):
    """
    The distribution associated with a dirac distribution at the origin
    """

    def __init__(self) -> None:
        pass

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return torch.where(points > 0.0, 1.0, 0.0)

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: float) -> CumulativeDistributionFunction:
        unused_variables(time)
        return self

    def boundaries(self) -> Dict[str, float]:
        return {
            'lower_bound': -1.0,
            'upper_bound':  1.0
        }
