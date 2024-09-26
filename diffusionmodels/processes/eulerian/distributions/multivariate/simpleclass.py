"""
diffusionmodels.distributions.multivariate.simpleclass
======================================================

Implements various simple multivariate distributions

Classes
-------
UniformSphere   : A uniformly distributed points on arbitrary-dimensional unit spheres
"""


from ..interfaces import Distribution

import torch

from typing import Callable, Tuple


class UniformSphere(Distribution):
    """
    A uniform distribution of points on an arbitrary-dimension unit sphere

    Private Attributes
    ------------------
    `_dimension: int`
        The dimension of the unit sphere

    `_device: torch.device`
        The device where all tensors are stored or created
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._num_times = 1
        self._device = torch.device('cpu')


    def dimension(self) -> Tuple[int, ...]:
        return (self._dimension,)

    def to(self, device: torch.device):
        self._device = device

    def at(self, time: torch.Tensor) -> Distribution:
        self._num_times = time.numel()
        return self

    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda _: torch.exp(

            torch.special.gammaln(0.5 * self._dimension)

        ) / (2.0 * torch.pi ** (0.5 * self._dimension))

    def sample(self, num_samples: int) -> torch.Tensor:

        points = torch.randn(
            size = (self._num_times, num_samples, self._dimension),
            device = self._device
        )

        return points / (torch.norm(points, dim = -1, keepdim = True) + 1e-6)
