"""
Implements the univariate uniform process and PDF.

Classes
-------
`Angle`
A uniformly distributed stochastic process with a fixed interval

`AngleDensity`
A uniform probability density function with a fixed interval
"""

from typing import Tuple

import torch

from ....interfaces import DensityFunction
from ..interfaces import CumulativeDistributionFunction


class Angle(CumulativeDistributionFunction):
    """
    The CDF of angle in axis angle representation that will generate uniform 2-Sphere
    """

    def __init__(self, data_type: torch.dtype = torch.float32) -> None:
        self._support = {"lower": 0.0, "upper": torch.pi}
        self._data_type = data_type
        self._density = AngleDensity(data_type)
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
        self._density.to(device)

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        self._density = self._density.at(self._time)
        return self

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        result = 0.5 * (1.0 - torch.cos(points))
        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result

    @property
    def gradient(self) -> DensityFunction:
        return self._density

    @property
    def support(self):
        return self._support


class AngleDensity(DensityFunction):
    """
    The PDF of angle in axis angle representation that will generate uniform 2-Sphere
    """

    def __init__(self, data_type: torch.dtype = torch.float32) -> None:
        self._data_type = data_type
        self._lower = 0.0
        self._upper = torch.pi
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
    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        result = 0.5 * torch.sin(points)
        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        result = 0.5 * torch.cos(points)
        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result
