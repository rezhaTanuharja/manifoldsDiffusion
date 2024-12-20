"""
Implements the univariate uniform probability density function.

Classes
-------
`ConstantUniform`
A uniformly distributed stochastic process with a fixed interval

`ConstantUniformDensity`
A uniform probability density function with a fixed interval
"""

from typing import Dict, Tuple

import torch

from ..interfaces import DensityFunction, StochasticProcess


class ConstantUniform(StochasticProcess):
    def __init__(
        self, support: Dict[str, float], data_type: torch.dtype = torch.float32
    ) -> None:
        self._support = support
        self._data_type = data_type
        self._density = ConstantUniformDensity(support, data_type)
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

    @property
    def dimension(self) -> Tuple[int, ...]:
        return self._density.dimension

    @property
    def density(self) -> DensityFunction:
        return self._density

    def sample(self, num_samples: int) -> torch.Tensor:
        lower_bound = self._support["lower"]
        upper_bound = self._support["upper"]

        return lower_bound + (upper_bound - lower_bound) * torch.rand(
            size=(*self._time.shape, num_samples, *self.dimension),
            dtype=self._data_type,
            device=self._device,
        )


class ConstantUniformDensity(DensityFunction):
    """
    A uniform probability density function defined on an interval
    """

    def __init__(
        self, support: Dict[str, float], data_type: torch.dtype = torch.float32
    ) -> None:
        """
        Construct a density object for a univariate uniform distribution

        Parameters
        ----------
        `support: Dict[str, float]`
        A dictionary with keys 'lower' and 'upper', defines the interval

        `data_type: torch.dtype = torch.float32`
        The data type of floating points
        """
        self._data_type = data_type
        self._lower = support["lower"]
        self._upper = support["upper"]
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> None:
        self._device = device
        self._time.to(device)

    def at(self, time: torch.Tensor):
        self._time = time
        self._time.to(self._device)
        return self

    @property
    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        lower_bound = torch.full_like(points, self._lower)
        upper_bound = torch.full_like(points, self._upper)

        values = 1.0 / (upper_bound - lower_bound)

        result = torch.where(
            (points < lower_bound) | (points > upper_bound),
            input=torch.zeros_like(points),
            other=values,
        )

        result = result.unsqueeze(0)
        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(
            input=points, dtype=self._data_type, device=self._device
        )

        result = result.unsqueeze(0)
        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result
