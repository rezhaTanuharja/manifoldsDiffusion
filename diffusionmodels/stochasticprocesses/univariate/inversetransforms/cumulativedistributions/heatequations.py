"""
Implements various CDFs based on the solutions of heat equations.

Classes
-------
`PeriodicHeatKernel`
The fundamental solution of heat diffusion in a periodic domain

`PeriodicCumulativeEnergy`
The spatial integration of `PeriodicHeatKernel` in `[0, pi]`
"""

from typing import Callable, Dict, Tuple

import torch

from ....interfaces import DensityFunction
from ..interfaces import CumulativeDistributionFunction


def standard_sum_weights(_: int, indices: torch.Tensor):
    """
    Compute the weight of terms in an infinite summation with standard summation

    Parameters
    ----------
    `_: int`
    The `alpha` for `(C, alpha)` summation, not used here

    `indices: torch.Tensor`
    The tensor of indices of the terms

    Returns
    -------
    `torch.Tensor`
    Tensor of ones with the same shape as `indices`
    """
    return torch.ones(size=indices.shape, dtype=indices.dtype, device=indices.device)


def cesaro_sum_weights(alpha: int, indices: torch.Tensor) -> torch.Tensor:
    """
    Compute the weight of terms in an infinite summation with Cesaro summation

    Parameters
    ----------
    `alpha: int`
    The `alpha` for `(C, alpha)` summation, see Cesaro higher order summations

    `indices: torch.Tensor`
    The tensor of indices of the terms

    Returns
    -------
    `torch.Tensor`
    Tensor of weights with the same shape as `indices`
    """
    return torch.binomial(
        torch.full_like(
            indices,
            fill_value=indices.numel(),
            dtype=indices.dtype,
            device=indices.device,
        ),
        indices,
    ) / torch.binomial(
        torch.full_like(
            indices,
            fill_value=indices.numel() + alpha,
            dtype=indices.dtype,
            device=indices.device,
        ),
        indices,
    )


class PeriodicHeatKernel(DensityFunction):
    """
    The fundamental solution of heat diffusion in a periodic domain
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
        alpha: int = 2,
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct a periodic heat kernel

        Parameters
        ----------
        `num_waves: int`
        The number of wave functions excluding the constant function

        `mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor]`
        A function representing the integration of the diffusion constant

        `alpha: int = 2`
        The parameter `alpha` in `(C, alpha)` summation

        `data_type`
        The data type of all tensor class attributes
        """
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

        self._alpha = alpha

        self._infinite_sum_weights = (
            standard_sum_weights if alpha == 0 else cesaro_sum_weights
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self._time = self._time.to(device)

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        return self

    @property
    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.full_like(
                self._time.unsqueeze(-1) + points,
                fill_value=1.0 / torch.pi,
                dtype=self._data_type,
                device=self._device,
            )

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points.unsqueeze(-1)

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
                    self._infinite_sum_weights(self._alpha, wave_numbers)
                    * temporal_components
                    * torch.cos(angles),
                    dim=-1,
                )
            )
        )

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.full_like(
                self._time.unsqueeze(-1) + points,
                fill_value=0.0,
                dtype=self._data_type,
                device=self._device,
            )

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points.unsqueeze(-1)

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
                * self._infinite_sum_weights(self._alpha, wave_numbers)
                * temporal_components
                * torch.sin(angles),
                dim=-1,
            )
        )


class PeriodicCumulativeEnergy(CumulativeDistributionFunction):
    """
    The spatial integration of `PeriodicHeatKernel` in `[0, pi]`
    """

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
        data_type: torch.dtype = torch.float32,
        alpha: int = 0,
    ):
        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement
        self._distribution = PeriodicHeatKernel(
            num_waves, mean_squared_displacement, alpha, data_type
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

        self._alpha = alpha

        self._infinite_sum_weights = (
            standard_sum_weights if alpha == 0 else cesaro_sum_weights
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self._time.to(device)

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        self._distribution = self._distribution.at(time)
        return self

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if self._num_waves == 0:
            return torch.zeros(self._time.shape).unsqueeze(-1) + points / torch.pi

        wave_numbers = torch.arange(
            1, self._num_waves + 1, dtype=self._data_type, device=self._device
        )
        wave_numbers = wave_numbers.reshape(
            *(1 for _ in points.shape),
            wave_numbers.numel(),
        )

        angles = wave_numbers * points.unsqueeze(-1)

        time = self._time.unsqueeze(-1).unsqueeze(-1)

        temporal_components = torch.exp(
            -self._mean_squared_displacement(time) * wave_numbers**2
        )

        return (
            1.0
            / torch.pi
            * (
                points
                + 2.0
                * torch.sum(
                    self._infinite_sum_weights(self._alpha, wave_numbers)
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
