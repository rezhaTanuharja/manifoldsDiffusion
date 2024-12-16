"""
Find root using the bisection method

Classes
-------
`Bisection`    : find root using the bisection method
"""

from typing import Callable, Tuple

import torch

from ..interfaces import RootFinder


class Bisection(RootFinder):
    def __init__(self, num_iterations: int):
        self._num_iterations = num_iterations

    def solve(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        target_values: torch.Tensor,
        interval: Tuple[float, float],
    ) -> torch.Tensor:
        lower_bound = torch.full_like(
            target_values, fill_value=interval[0], dtype=target_values.dtype
        )
        upper_bound = torch.full_like(
            target_values, fill_value=interval[1], dtype=target_values.dtype
        )

        for _ in range(self._num_iterations):
            midpoint = 0.5 * (lower_bound + upper_bound)

            function_values = function(midpoint)

            lower_bound = torch.where(
                function_values < target_values, midpoint, lower_bound
            )

            upper_bound = torch.where(
                function_values >= target_values, midpoint, upper_bound
            )

        return lower_bound
