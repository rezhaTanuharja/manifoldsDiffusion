"""
Implements the univariate uniform process and PDF.

Classes
-------
`Uniform`
A uniformly distributed stochastic process with a fixed interval

`UniformDensity`
A uniform probability density function with a fixed interval
"""

from typing import Dict, Tuple

import torch

from ..interfaces import DensityFunction, StochasticProcess


class Uniform(StochasticProcess):
    """
    A uniform stochastic process defined on a fixed interval
    """

    def __init__(
        self,
        support: Dict[str, float],
        device: torch.device = torch.device("cpu"),
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct a univariate uniform process

        Parameters
        ----------
        `support: Dict[str, float]`
        A dictionary containing keys `lower` and `upper` defining the interval where values may be non-zero

        `device: torch.device = torch.device("cpu")`
        The hardware where tensor attributes reside

        `data_type: torch.dtype = torch.float32`
        The type of floating point
        """
        assert "lower" in support.keys(), "Missing lower bound of the support interval"
        assert "upper" in support.keys(), "Missing upper bound of the support interval"

        self._support = support
        self._data_type = data_type
        self._density = UniformDensity(support, device, data_type)
        self._time = torch.tensor(
            [
                0.0,
            ],
            device=device,
            dtype=data_type,
        )
        self._device = device

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
            size=(*self._time.shape, num_samples),
            dtype=self._data_type,
            device=self._device,
        )


class UniformDensity(DensityFunction):
    """
    A uniform probability density function defined on an interval
    """

    def __init__(
        self,
        support: Dict[str, float],
        device: torch.device = torch.device("cpu"),
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct a density object for a univariate uniform distribution

        Parameters
        ----------
        `support: Dict[str, float]`
        A dictionary with keys 'lower' and 'upper', defines the interval

        `device: torch.device = torch.device("cpu")`
        The hardware where tensor attributes reside

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
            device=device,
            dtype=data_type,
        )
        self._device = device

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
        lower_bound = torch.full_like(points, self._lower)
        upper_bound = torch.full_like(points, self._upper)

        values = 1.0 / (upper_bound - lower_bound)

        result = torch.where(
            (points < lower_bound) | (points > upper_bound),
            input=torch.zeros_like(points),
            other=values,
        )

        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(
            input=points, dtype=self._data_type, device=self._device
        )

        result = result.repeat((*self._time.shape, *(1 for _ in result.shape[1:])))

        return result
