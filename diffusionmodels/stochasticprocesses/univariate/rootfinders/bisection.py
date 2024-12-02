from .. import RootFinder

import jax.numpy as jnp

from typing import Callable, Tuple


class Bisection(RootFinder):

    def __init__(self, num_iterations: int):

        self._num_iterations = num_iterations

    def solve(
        self,
        function: Callable[[jnp.ndarray], jnp.ndarray],
        target_values: jnp.ndarray,
        interval: Tuple[float, float],
    ) -> jnp.ndarray:

        lower_bound = jnp.full_like(target_values, fill_value=interval[0])
        upper_bound = jnp.full_like(target_values, fill_value=interval[1])

        for _ in range(self._num_iterations):

            midpoint = 0.5 * (lower_bound + upper_bound)

            function_values = function(midpoint)

            lower_bound = jnp.where(
                function_values < target_values, midpoint, lower_bound
            )

            upper_bound = jnp.where(
                function_values >= target_values, midpoint, upper_bound
            )

        return lower_bound
