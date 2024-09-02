"""
scorefunctions.baseclass
========================

This module implements simple classes of scorefunctions

Classes
-------
"""


import torch
from .baseclass import RelativeDirectionCalculator

from ..manifolds import Manifold


class Direction(RelativeDirectionCalculator):

    def __init__(self, manifold: Manifold):
        super().__init__(manifold)


    # def get_direction(
    #     self,
    #     X: torch.Tensor, X_ref: torch.Tensor,
    #     t: float,
    # ) -> torch.Tensor:
    def get_direction(
        self,
        origin: torch.Tensor, destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the direction, dX, to update X(t)

        Parameters
        ----------
        X : torch.Tensor
            The current spatial location, i.e., X(t)

        t : float
            The current time

        Returns
        -------
        torch.Tensor
            The direction of update, dX
        """
        direction = self._manifold.log(origin, destination)
        return torch.einsum('i..., i... -> i...', scale, direction)
        # return 1.0 / (t) * self._manifold.log(X, X_ref)
