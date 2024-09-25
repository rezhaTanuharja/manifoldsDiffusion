"""
diffusionmodels.distributions.univariate.inversion.simpleclass
==============================================================

Implements various simple inversion methods

Classes
-------
Bisection
    The inversion method via a bisection root-finder
"""


from typing import Callable, Dict
import torch

from .baseclass import InversionMethod


class Bisection(InversionMethod):
    """
    An inversion method using the bisection root-finder

    Private Attributes
    ------------------
    `_num_iterations: int`
        The number of iteration to perform to find the roots

    `_device: torch.device`
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
        function: Callable[[torch.Tensor], torch.Tensor],
        search_range: Dict[str, float]
    ) -> torch.Tensor:

        lower_bound = torch.full_like(values, search_range['lower_bound'], device = self._device)
        upper_bound = torch.full_like(values, search_range['upper_bound'], device = self._device)

        for _ in range(self._num_iterations):

            midpoint = 0.5 * (lower_bound + upper_bound)

            function_values = function(midpoint)

            lower_bound = torch.where(function_values <  values, midpoint, lower_bound)
            upper_bound = torch.where(function_values >= values, midpoint, upper_bound)

        return 0.5 * (lower_bound + upper_bound)
