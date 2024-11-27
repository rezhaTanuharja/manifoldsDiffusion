"""
Implement univariate stochastic processes that are defined by their CDFs.

Classes
-------
`InverseTransform`
A stochastic process defined by a CDF and sampled using inverse transform method
"""


from .. import DensityFunction, CumulativeDistributionFunction, StochasticProcess

from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from jaxopt import Bisection


class InverseTransform(StochasticProcess):


    def __init__(self, distribution: CumulativeDistributionFunction) -> None:
        self._distribution = distribution


    def to(self, device: jax.Device) -> None:
        self._distribution.to(device)


    def dimension(self) -> Tuple[int, ...]:
        return (1,)


    def density(self) -> DensityFunction:
        return self._distribution.gradient()


    # def sample(
    #     self, num_samples: int, times: Optional[jnp.ndarray]
    # ) -> jnp.ndarray:
    #
    #
    #
