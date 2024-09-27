"""
lagrangian.scorefunctions.simpleclass
=====================================

This module implements simple classes of scorefunctions

Classes
-------
Geodesic        : A score function in the form of a scaled geodesic distance
"""


import torch
from .interfaces import Direction

from ...manifolds import Manifold


class Geodesic(Direction):

    def __init__(self, manifold: Manifold):
        self._manifold = manifold

    def to(self, device: torch.device) -> None:
        self._manifold.to(device)

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
