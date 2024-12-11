"""
Implements various `CumulativeDistributionFunction` in the form of polynomials.

Classes
-------
`Linear`
A class of CDF with linear function
"""

from typing import Dict

import torch

from ....utilities.warningsuppressors import unused_variables
from ..interfaces import CumulativeDistributionFunction


class Linear(CumulativeDistributionFunction):
    def __init__(self, support: Dict[str, float]) -> None:
        self._lower = support["lower"]
        self._upper = support["upper"]

    def to(self, device: torch.device) -> None:
        unused_variables(device)

    def at(self, time: torch.Tensor):
        unused_variables(time)
        return self

    def __call__(
        self,
        points: torch.Tensor,
        times: torch.Tensor = torch.tensor(
            [
                0.0,
            ]
        ),
    ) -> torch.Tensor:
        unused_variables(times)

        return torch.clip(
            (points - self._lower) / (self._upper - self._lower),
            min=0.0,
            max=1.0,
        )

    # def gradient(self) -> DensityFunction:
    #     pass
