"""
distributions.functions.simpleclass
===================================

Implements various simple PDFs and CDFs

Classes
-------
Linear
    The CDF of a uniform distribution (useless, just to test the interface)
"""


from diffusionmodels.utilities.warningsuppressors import unused_variables
from .baseclass import CumulativeDistributionFunction

from typing import Dict

import torch


class Linear(CumulativeDistributionFunction):

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: float) -> CumulativeDistributionFunction:
        unused_variables(time)
        return self

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self._upper_bound - self._lower_bound) * (
            -self._lower_bound + points
        )

    def boundaries(self) -> Dict[str, float]:
        return {
            'lower_bound': self._lower_bound,
            'upper_bound': self._upper_bound,
        }
