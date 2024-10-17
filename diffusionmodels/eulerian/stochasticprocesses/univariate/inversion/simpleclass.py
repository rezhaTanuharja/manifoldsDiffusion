"""
eulerian.stochasticprocesses.univariate.inversion.simpleclass
=============================================================

Implements various simple inversion methods

Classes
-------
Bisection
    An iterative root-finder via bisection
"""


from typing import Callable, Dict, Optional
import torch

from .interfaces import InversionMethod


class Bisection(InversionMethod):
    """
    An iterative root-finder via bisection
    """

    def __init__(
        self,
        num_iterations: int,
        pretransform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        """
        Parameters
        ----------
        `num_iterations: int`
            The number of iteration to perform to find the roots

        `pretransform: Optional[Callable[[torch.Tensor], torch.Tensor]]`
            When specified, the inverse of pretransform is applied to the results of `solve`
        """
        self._num_iterations = num_iterations
        self._device = torch.device('cpu')

        self._pretransform = (lambda points: points) if pretransform == None else pretransform

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

            function_values = function(self._pretransform(midpoint))

            lower_bound = torch.where(function_values <  values, midpoint, lower_bound)
            upper_bound = torch.where(function_values >= values, midpoint, upper_bound)

        return lower_bound
