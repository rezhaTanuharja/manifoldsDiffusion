"""
diffusionmodels.distributions.univariate.functions.periodic
===========================================================

Provides solutions to PDEs with periodic boundary conditions

Classes
-------
HeatKernel
    The solution to a heat equation with periodic boundary conditions
"""

from ....utilities.warningsuppressors import unused_variables
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

    `_time: torch.Tensor`
        Represent the time at which the distribution is evaluated
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._time = torch.tensor([0.0,])

    def cumulative(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[None, :, *[None for _ in range(points.dim() - 1)]]

        time = self._time[:, *[None for _ in range(points.dim())]]

        angles = wave_numbers * points.unsqueeze(1)

        temporal_components = 1.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers ** 2
        ) / wave_numbers

        return 0.5 / torch.pi * (torch.pi + points) + torch.sum(
            temporal_components * torch.sin(angles),
            dim = 1
        )

    def density(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[None, :, *[None for _ in range(points.dim() - 1)]]

        time = self._time[:, *[None for _ in range(points.dim())]]

        angles = wave_numbers * points.unsqueeze(1)

        temporal_components = 1.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers ** 2
        )

        return 0.5 / torch.pi + torch.sum(
            temporal_components * torch.cos(angles),
            dim = 1
        )

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def at(self, time: torch.Tensor) -> DistributionFunction:
        self._time = time
        return self

    def support(self) -> Dict[str, float]:
        return {
            'lower_bound': -torch.pi,
            'upper_bound':  torch.pi
        }
