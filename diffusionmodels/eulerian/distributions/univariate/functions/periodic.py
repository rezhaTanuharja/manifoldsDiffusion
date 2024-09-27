"""
diffusionmodels.distributions.univariate.functions.periodic
===========================================================

Provides solutions to PDEs with periodic boundary conditions

Classes
-------
HeatKernel
    The solution to a heat equation with periodic boundary conditions
"""

from .interfaces import CumulativeDistributionFunction
from typing import Dict, Callable
import torch


class HeatKernel(CumulativeDistributionFunction):
    """
    The solution to a heat equation with periodic boundary conditions

    Private Attributes
    ------------------
    `_num_waves: int`
        The number of wave functions in the solution

    `_mean_squared_displacement: Callable[[float], float]`
        The integral of the diffusion coefficient over time, as a function

    `_time: torch.Tensor`
        Represent the time(s) at which the CDF is evaluated
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._time = torch.tensor([0.0,])

    def to(self, device: torch.device) -> None:
        self._time = self._time.to(device)

    def at(self, time: torch.Tensor) -> CumulativeDistributionFunction:
        self._time = time
        return self

    def __call__(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[None, :, *[None for _ in range(points.dim() - 1)]]

        points = points - torch.sin(points)

        angles = wave_numbers * points.unsqueeze(1)

        time = self._time[:, *[None for _ in range(points.dim())]]

        temporal_components = 2.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers ** 2
        ) / wave_numbers

        return 1.0 / torch.pi * points + torch.sum(
            temporal_components * torch.sin(angles),
            dim = 1
        )

    def gradient(self, points: torch.Tensor) -> torch.Tensor:

        wave_numbers = torch.arange(start = 1, end = self._num_waves + 1, device = points.device)
        wave_numbers = wave_numbers[None, :, *[None for _ in range(points.dim() - 1)]]

        angles = wave_numbers * (points - torch.sin(points)).unsqueeze(1)

        time = self._time[:, *[None for _ in range(points.dim())]]

        temporal_components = 2.0 / torch.pi * torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers ** 2
        )

        return (1 - torch.cos(points)) * (
            1.0 / torch.pi + torch.sum(
                temporal_components * torch.cos(angles),
                dim = 1
            )
        )

    def support(self) -> Dict[str, float]:
        return {
            'lower_bound': 0.0,
            'upper_bound': torch.pi
        }