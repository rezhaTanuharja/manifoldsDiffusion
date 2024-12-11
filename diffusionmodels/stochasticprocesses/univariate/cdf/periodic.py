from typing import Callable

import torch

from ....utilities.warningsuppressors import unused_variables
from .. import CumulativeDistributionFunction


class HeatKernel(CumulativeDistributionFunction):
    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement

    def to(self, device: torch.device):
        unused_variables(device)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(points)
