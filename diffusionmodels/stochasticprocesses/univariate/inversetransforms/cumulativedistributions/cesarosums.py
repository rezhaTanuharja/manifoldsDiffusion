"""
Implements various CDFs based on Cesaro higher order summation.

Classes
-------
`CesaroSumDensity`
The Cesaro higher order summation of the `PeriodicHeatKernel`

`CesaroSum`
The spatial integral of `CesaroSumDensity`
"""

from typing import Callable, Dict

import torch

from ....interfaces import DensityFunction
from ..interfaces import CumulativeDistributionFunction


class CesaroSumDensity(DensityFunction):
    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
        data_type: torch.dtype = torch.float32,
    ):
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

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape[:-1]),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points

        time = self._time.unsqueeze(-1).unsqueeze(-1)

        temporal_components = torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers**2
        )

        return (
            1.0
            / torch.pi
            * (
                1.0
                + 2.0
                * torch.sum(
                    torch.binomial(
                        torch.tensor(
                            [
                                self._num_waves - 2,
                            ],
                            dtype=self._data_type,
                        )
                        .repeat((self._num_waves,))
                        .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                        wave_numbers,
                    )
                    / torch.binomial(
                        torch.tensor(
                            [
                                self._num_waves,
                            ],
                            dtype=self._data_type,
                        )
                        .repeat((self._num_waves,))
                        .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                        wave_numbers,
                    )
                    * temporal_components
                    * torch.cos(angles),
                    dim=-1,
                )
            )
        )

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.full_like(
                self._time.unsqueeze(-1) + points.squeeze(-1), fill_value=0.0
            )

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape[:-1]),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points

        time = self._time.unsqueeze(-1).unsqueeze(-1)

        temporal_components = torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers**2
        )

        return (
            1.0
            / torch.pi
            * -2.0
            * torch.sum(
                wave_numbers
                * torch.binomial(
                    torch.tensor(
                        [
                            self._num_waves - 2,
                        ],
                        dtype=self._data_type,
                    )
                    .repeat((self._num_waves,))
                    .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                    wave_numbers,
                )
                / torch.binomial(
                    torch.tensor(
                        [
                            self._num_waves,
                        ],
                        dtype=self._data_type,
                    )
                    .repeat((self._num_waves,))
                    .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                    wave_numbers,
                )
                * temporal_components
                * torch.sin(angles),
                dim=-1,
            )
        )


class CesaroSum(CumulativeDistributionFunction):
    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
        data_type: torch.dtype = torch.float32,
    ):
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._distribution = CesaroSumDensity(
            num_waves, mean_squared_displacement, data_type
        )
        self._data_type = data_type
        self._device = torch.device("cpu")
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
            device=torch.device("cpu"),
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self._time.to(device)

    def __call__(self, points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return (
                torch.zeros(times.shape).unsqueeze(-1) + points.squeeze(-1) / torch.pi
            )

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape[:-1]),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points

        time = times.unsqueeze(-1).unsqueeze(-1)

        temporal_components = torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers**2
        )

        return (
            1.0
            / torch.pi
            * (
                points.squeeze(-1)
                + 2.0
                * torch.sum(
                    torch.binomial(
                        torch.tensor(
                            [
                                self._num_waves - 2,
                            ],
                            dtype=self._data_type,
                        )
                        .repeat((self._num_waves,))
                        .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                        wave_numbers,
                    )
                    / torch.binomial(
                        torch.tensor(
                            [
                                self._num_waves,
                            ],
                            dtype=self._data_type,
                        )
                        .repeat((self._num_waves,))
                        .reshape((*(1 for _ in points.shape[:-1]), self._num_waves)),
                        wave_numbers,
                    )
                    / wave_numbers
                    * temporal_components
                    * torch.sin(angles),
                    dim=-1,
                )
            )
        )

    @property
    def gradient(self) -> DensityFunction:
        return self._distribution

    @property
    def support(self) -> Dict[str, float]:
        return {"lower": 0.0, "upper": torch.pi}
