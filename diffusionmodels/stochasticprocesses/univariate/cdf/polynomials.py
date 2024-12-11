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
        self._support = support

    def to(self, device: torch.device) -> None:
        unused_variables(device)

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
            (points - self._support["lower"])
            / (self._support["upper"] - self._support["lower"]),
            min=0.0,
            max=1.0,
        )
