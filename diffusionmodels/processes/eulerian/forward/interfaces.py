
from ....manifolds import Manifold
from ..distributions import Distribution
from typing import Callable, Tuple

import torch


class RandomVector(Distribution):

    def __init__(
        self,
        magnitude_distribution: Distribution,
        direction_distribution: Distribution
    ):
        self._magnitude_distribution = magnitude_distribution
        self._direction_distribution = direction_distribution

    def dimension(self) -> Tuple[int, ...]:
        return self._direction_distribution.dimension()

    def to(self, device):
        self._magnitude_distribution = self._magnitude_distribution.to(device)
        self._direction_distribution = self._direction_distribution.to(device)

    def at(self, time):
        self._magnitude_distribution = self._magnitude_distribution.at(time)
        self._direction_distribution = self._direction_distribution.at(time)
        return self

    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:

        def combined_density(points):

            magnitude = torch.norm(points, dim = -1, keepdim =  True)
            direction = points / magnitude

            return (
                self._magnitude_distribution.density_function()(magnitude)
                *
                self._direction_distribution.density_function()(direction)
            )

        return combined_density

    def sample(self, num_samples: int) -> torch.Tensor:

        return torch.einsum(
            'ij..., ij... -> ij...',
            self._magnitude_distribution.sample(num_samples),
            self._direction_distribution.sample(num_samples)
        )



class RandomFlow(Distribution):

    def __init__(
        self,
        manifold: Manifold,
        vector_distribution: Distribution
    ) -> None:

        if manifold.tangent_dimension() != vector_distribution.dimension():
            raise ValueError('Manifold and Vector Distribution are incompatible')

        self._manifold = manifold
        self._vector_distribution = vector_distribution

    def dimension(self) -> Tuple[int, ...]:
        return self._manifold.dimension()

    def to(self, device: torch.device) -> None:
        self._vector_distribution.to(device)

    def at(self, time: torch.Tensor) -> Distribution:
        self._vector_distribution.at(time)
        return self

    def density_function(self, initial_value: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:

        return lambda points: self._vector_distribution.density_function()(
            self._manifold.log(initial_value, points)
        )

    def sample(self, initial_value: torch.Tensor, num_samples: int):

        # return self._vector_distribution.sample(num_samples)

        return self._manifold.exp(
            initial_value,
            self._vector_distribution.sample(num_samples)
        )
