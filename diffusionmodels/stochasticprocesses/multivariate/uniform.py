from typing import Tuple

import torch

from ..interfaces import DensityFunction, StochasticProcess


class UniformSphereDensity(DensityFunction):
    """
    The uniform density on an n-dimensional sphere
    """

    def __init__(self, dimension: int, data_type: torch.dtype = torch.float32):
        self._dimension = dimension
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
            device=torch.device("cpu"),
        )
        self._device = torch.device("cpu")
        self._data_type = data_type

    def to(self, device: torch.device) -> None:
        self._time = self._time.to(device)
        self._device = device

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        return self

    @property
    def dimension(self) -> Tuple[int]:
        return (self._dimension,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            size=(*self._time.shape, points.shape[-2]),
            dtype=self._data_type,
            device=self._device,
        ) / (
            2.0
            * torch.pi ** (0.5 * (self._dimension + 1))
            / torch.lgamma(torch.tensor(0.5 * (self._dimension + 1))).exp()
        )

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            size=(*self._time.shape, *points.shape[-2:]),
            dtype=self._data_type,
            device=self._device,
        )


class UniformSphere(StochasticProcess):
    """
    The uniform stochastic process on an n-dimensional sphere
    """

    def __init__(self, dimension: int, data_type: torch.dtype = torch.float32):
        self._dimension = dimension
        self._density = UniformSphereDensity(dimension, data_type)
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
            device=torch.device("cpu"),
        )
        self._device = torch.device("cpu")
        self._data_type = data_type

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        self._density = self._density.at(time)
        return self

    def to(self, device: torch.device) -> None:
        self._time = self._time.to(device)
        self._density.to(device)
        self._device = device

    @property
    def dimension(self) -> Tuple[int]:
        return (self._dimension,)

    @property
    def density(self) -> DensityFunction:
        return self._density

    def sample(self, num_samples: int) -> torch.Tensor:
        samples = torch.randn(
            size=(*self._time.shape, num_samples, self._dimension),
            dtype=self._data_type,
            device=self._device,
        )
        samples = samples / torch.norm(samples, keepdim=True)

        return samples
