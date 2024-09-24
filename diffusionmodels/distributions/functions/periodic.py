"""
distributions.functions.periodic
================================

Implements various simple PDFs and CDFs that is a solution to a PDE with periodic boundary conditions

Classes
-------
HeatKernel
    The solution to a heat equation with periodic boundary conditions
"""

from ...utilities.warningsuppressors import unused_variables
from .baseclass import DistributionFunction
from typing import Dict, Callable
import torch


class HeatKernel(DistributionFunction):
    """
    The solution to a heat equation with periodic boundary conditions

    Private Attributes
    ------------------
    `_num_waves: int`
        The number of wave functions in the solution

    `_mean_squared_displacement: Callable[[float], float]`
        The integral of the diffusion coefficient over time, as a function

    `_time: float`
        Represent the current time of the function
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[float], float]
    ) -> None:
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._time = 0.0

    def cumulative(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[:, *[None for _ in range(points.dim())]]

        angles = wave_numbers * points[None, ...]

        temporal_components = 1.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(self._time) * wave_numbers ** 2
        ) / wave_numbers

        return 0.5 / torch.pi * (torch.pi + points) + torch.sum(
            temporal_components * torch.sin(angles),
            dim = 0
        )

    def density(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[:, *[None for _ in range(points.dim())]]

        angles = wave_numbers * points[None, ...]

        temporal_components = 1.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(self._time) * wave_numbers ** 2
        )

        return 0.5 / torch.pi + torch.sum(
            temporal_components * torch.cos(angles),
            dim = 0
        )

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: float) -> DistributionFunction:
        self._time = time
        return self

    def boundaries(self) -> Dict[str, float]:
        return {
            'lower_bound': -torch.pi,
            'upper_bound':  torch.pi
        }
