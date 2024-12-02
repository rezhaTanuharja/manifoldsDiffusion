from ....utilities.warningsuppressors import unused_variables

from .. import CumulativeDistributionFunction

import jax
import jax.numpy as jnp

from typing import Callable


class HeatKernel(CumulativeDistributionFunction):

    def __init__(
        self,
        num_waves: int,
        mean_squared_displacement: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:

        self._num_waves = num_waves
        self._mean_squared_displacement = mean_squared_displacement

    def to(self, device: jax.Device):
        unused_variables(device)

    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:

        return jnp.zeros_like(points)
