"""
Implement univariate stochastic processes that are defined by their CDFs.

Classes
-------
`InverseTransform`
A stochastic process defined by a CDF and sampled using inverse transform method
"""


from .. import DensityFunction, StochasticProcess
from univariate import CumulativeDistributionFunction, RootFinder

from typing import Tuple

import jax
import jax.numpy as jnp

import numpy as np


class InverseTransform(StochasticProcess):


    def __init__(
        self,
        distribution: CumulativeDistributionFunction,
        root_finder: RootFinder
    ) -> None:
        self._distribution = distribution
        self._root_finder = root_finder


    def to(self, device: jax.Device) -> None:
        self._distribution.to(device)


    def dimension(self) -> Tuple[int, ...]:
        return (1,)


    def density(self) -> DensityFunction:
        return self._distribution.gradient()


    def sample(
        self, num_samples: int, times: jnp.ndarray = jnp.array([0.0,])
    ) -> jnp.ndarray:

        seed = np.random.randint(0, 2 ** 32 - 1)
        key = jax.random.PRNGKey(seed)

        return self._root_finder.solve(

            function = lambda points: self._distribution(points, times),

            target_values = jax.random.uniform(
                key = key, shape = (times.shape[-1], num_samples)
            ),

            interval = (
                self._distribution.support()['lower'],
                self._distribution.support()['upper']
            )

        )
