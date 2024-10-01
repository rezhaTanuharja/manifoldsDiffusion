"""
eulerian.stochasticprocesses.univariate.inversion.simpleclass
=============================================================

Implements various simple inversion methods

Classes
-------
Bisection
    An iterative root-finder via bisection

Newton
    An iterative root-finder that requires computing gradients

Secant
    A Newton-like method that does not requires computing gradients
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


class Secant(InversionMethod):
    """
    A Newton-like method that does not requires computing gradients
    """

    def __init__(
        self,
        max_iter: int,
        tolerance: float,
    ) -> None:
        """
        Parameters
        ----------
        `max_iterations: int`
            The number of maximum iteration to perform to find the roots

        `tolerance: float`
            The accepted level of error
        """
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._device = torch.device('cpu')

    def to(self, device: torch.device) -> None:
        self._device = device

    def solve(
        self,
        values: torch.Tensor,
        function: Callable[[torch.Tensor], torch.Tensor],
        search_range: Dict[str, float]
    ) -> torch.Tensor:

        prev_guess = torch.full_like(values, search_range['lower_bound'])

        guess = torch.full_like(
            values, 0.3 * search_range['upper_bound'] + 0.7 * search_range['lower_bound']
        )

        for _ in range(self._max_iter):

            error = function(guess) - values

            converged = torch.abs(error) < self._tolerance
            if torch.all(converged):
                break

            gradient = torch.clip(
                (function(guess) - function(prev_guess)) / (guess - prev_guess),
                min = 0.75 / (search_range['upper_bound'] - search_range['lower_bound'])
            )

            prev_guess = guess.clone()
            guess = guess - error / gradient

        return guess
