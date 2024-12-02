"""
Implements various `CumulativeDistributionFunction` in the form of polynomials.

Classes
-------
`Linear`
A class of CDF with linear function
"""

from ....utilities.warningsuppressors import unused_variables
from .. import CumulativeDistributionFunction

from typing import Dict

import jax
import jax.numpy as jnp


class Linear(CumulativeDistributionFunction):

    def __init__(self, support: Dict[str, float]) -> None:
        self._support = support

    def to(self, device: jax.Device) -> None:
        unused_variables(device)

    def __call__(
        self,
        points: jnp.ndarray,
        times: jnp.ndarray = jnp.array(
            [
                0.0,
            ]
        ),
    ) -> jnp.ndarray:

        unused_variables(times)

        return jnp.clip(
            (points - self._support["lower"])
            / (self._support["upper"] - self._support["lower"]),
            min=0.0,
            max=1.0,
        )
