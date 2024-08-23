"""
diffusionmodels.scorefunctions.baseclass
========================================

This module implements simple classes of scorefunctions

Classes
-------
"""


import torch
from .baseclass import RelativeDirectionCalculator

from ..manifolds import Manifold


class DirectToReference(RelativeDirectionCalculator):

    def __init__(self, manifold: Manifold, X_ref: torch.Tensor, t_ref: torch.Tensor):
        super().__init__(manifold, X_ref, t_ref)


    def get_direction(
        self,
        X: torch.Tensor,
        t: float,
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
        return 1.0 / (self.t_ref - t) * self.manifold.log(X, self.X_ref)
        # (self.X_ref - X) 1.0 / (self.t_ref - t)
