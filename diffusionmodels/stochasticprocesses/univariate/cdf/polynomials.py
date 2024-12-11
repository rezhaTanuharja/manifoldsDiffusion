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
from ..interfaces import CumulativeDistributionFunction, DensityFunction
from ..uniform import Uniform


class Linear(CumulativeDistributionFunction):
    def __init__(
        self, support: Dict[str, float], data_type: torch.dtype = torch.float32
    ) -> None:
        self._density = Uniform(support=support, data_type=data_type)
        self._support = support
        self._data_type = data_type

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
            (points - self.support["lower"])
            / (self.support["upper"] - self.support["lower"]),
            min=0.0,
            max=1.0,
        )

    @property
    def gradient(self) -> DensityFunction:
        return self._density

    @property
    def support(self) -> Dict[str, float]:
        return self._support
