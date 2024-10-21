"""
stochasticprocesses.multivariate.simpleclass
============================================

Implements various simple multivariate distributions

Classes
-------
UniformSphere   : A uniformly distributed points on arbitrary-dimensional unit spheres
"""


from ..interfaces import StochasticProcess
from typing import Tuple

import torch



class UniformSphere(StochasticProcess):
    """
    A uniform distribution of points on an arbitrary-dimension unit sphere
    """

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        `dimension: int`
            The dimension of the unit sphere
        """
        self._dimension = dimension
        self._num_times = 1
        self._device = torch.device('cpu')


    def dimension(self) -> Tuple[int, ...]:
        return (self._dimension,)

    def to(self, device: torch.device) -> None:
        self._device = device

    def at(self, time: torch.Tensor) -> StochasticProcess:
        self._num_times = time.numel()
        return self

    def density(self, points: torch.Tensor) -> torch.Tensor:
        return torch.full_like(
            points,
            torch.exp(
                torch.special.gammaln(0.5 * self._dimension)
            ) / (2.0 * torch.pi ** (0.5 * self._dimension))
        )

    def score_function(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            size = points.shape + self.dimension()
        )

    def sample(self, num_samples: int) -> torch.Tensor:

        points = torch.randn(
            size = (self._num_times, num_samples, self._dimension),
            device = self._device
        )

        return points / torch.clip(input = torch.norm(points, dim = -1, keepdim = True), min = 1e-6)
