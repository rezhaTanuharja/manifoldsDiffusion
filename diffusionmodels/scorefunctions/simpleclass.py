"""
diffusionmodels.scorefunctions.baseclass
========================================

This module implements simple classes of scorefunctions

Classes
-------
"""


import torch
from .baseclass import RelativeDirectionCalculator


class DirectToReference(RelativeDirectionCalculator):

    def __init__(self, X_ref, t_ref):
        super().__init__(X_ref, t_ref)


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
        return (self.X_ref - X) / (self.t_ref - t)
