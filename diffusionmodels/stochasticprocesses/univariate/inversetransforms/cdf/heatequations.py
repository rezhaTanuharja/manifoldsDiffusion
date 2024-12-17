from typing import Callable

import torch

from ....interfaces import DensityFunction


class PeriodicHeatKernel(DensityFunction):
    """
    The fundamental solution to the heat equation in a periodic domain.
    The domain is `[-pi, pi]` but folded to be `[0, pi]` due to symmetry.
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct a periodic heat kernel density function.

        Parameters
        ----------
        `num_waves: int`
        The number of cosine functions to approximate the solution

        `mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor]`
        A function representing the integral of the diffusion coefficient over time

        `data_type: torch.dtype = torch.float32`
        The tensor data type for input and output
        """
        assert num_waves >= 0

        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._data_type = data_type
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> None:
        self._device = device
        self._time = self._time.to(device)

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        return self

    @property
    def dimension(self):
        return (1,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.full_like(
                self._time.unsqueeze(-1) + points.squeeze(-1), fill_value=1.0 / torch.pi
            )

        wave_numbers = 1 + torch.arange(self._num_waves, device=self._device)
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape[:-1]),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points

        time = self._time.unsqueeze(-1).unsqueeze(-1)

        temporal_components = torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers**2
        )

        return 1.0 / torch.pi + 2.0 / torch.pi * torch.sum(
            temporal_components * torch.cos(angles), dim=-1
        )

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.full_like(
                self._time.unsqueeze(-1) + points.squeeze(-1), fill_value=0.0
            )

        wave_numbers = 1 + torch.arange(self._num_waves, device=self._device)
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape[:-1]),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points

        time = self._time.unsqueeze(-1).unsqueeze(-1)

        temporal_components = (
            torch.exp(-self._mean_squared_displacement(time) * wave_numbers**2)
            * wave_numbers
        )

        return (
            -2.0 / torch.pi * torch.sum(temporal_components * torch.sin(angles), dim=-1)
        )
