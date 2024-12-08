"""
Find root using the bisection method

Classes
-------
`Bisection`    : find root using the bisection method
"""

from .. import RootFinder

import torch

from typing import Callable, Tuple


class Bisection(RootFinder):

    def __init__(self, num_iterations: int):
        self._num_iterations = num_iterations

    def solve(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        target_values: torch.Tensor,
        interval: Tuple[float, float],
    ) -> torch.Tensor:

        lower_bound = torch.full_like(target_values, fill_value=interval[0])
        upper_bound = torch.full_like(target_values, fill_value=interval[1])

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
