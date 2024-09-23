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

from ..functions import CumulativeDistributionFunction
from .baseclass import InversionMethod


class Bisection(InversionMethod):
    """
    An inversion method using the bisection root-finder

    Private Attributes
    ------------------
    `num_iterations: int`
        The number of iteration to perform to find the roots

    `device: torch.device`
        The device where all tensors are located
    """

    def __init__(self, num_iterations: int) -> None:
        self._num_iterations = num_iterations
        self._device = torch.device('cpu')

    def to(self, device: torch.device) -> None:
        self._device = device

    def solve(
        self,
        values: torch.Tensor,
        function: CumulativeDistributionFunction
    ) -> torch.Tensor:

        boundaries = function.boundaries()

        lower_bound = torch.full_like(values, boundaries['lower_bound'], device = self._device)
        upper_bound = torch.full_like(values, boundaries['upper_bound'], device = self._device)

        for _ in range(self._num_iterations):

            midpoint = 0.5 * (lower_bound + upper_bound)

            function_values = function.evaluate(midpoint)

            lower_bound = torch.where(function_values <  values, midpoint, lower_bound)
            upper_bound = torch.where(function_values >= values, midpoint, upper_bound)

        return 0.5 * (lower_bound + upper_bound)
