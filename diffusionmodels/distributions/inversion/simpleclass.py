"""
distributions.inversion.simpleclass
===================================

Implements various simple inversion methods

Classes
-------
Bisection
    The inversion method via a bisection root-finder
"""


import torch

from diffusionmodels.utilities.warningsuppressors import unused_variables
from ..functions import CumulativeDistributionFunction
from .baseclass import InversionMethod


class Bisection(InversionMethod):

    def __init__(self, num_iterations: int) -> None:
        self._num_iterations = num_iterations

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def solve(
        self,
        values: torch.Tensor,
        function: CumulativeDistributionFunction
    ) -> torch.Tensor:

        boundaries = function.boundaries()

        lower_bound = torch.full_like(values, boundaries['lower_bound'])
        upper_bound = torch.full_like(values, boundaries['upper_bound'])

        for _ in range(self._num_iterations):

            midpoint = 0.5 * (lower_bound + upper_bound)

            function_values = function.evaluate(midpoint)

            lower_bound = torch.where(function_values <  values, midpoint, lower_bound)
            upper_bound = torch.where(function_values >= values, midpoint, upper_bound)

        return 0.5 * (lower_bound + upper_bound)
