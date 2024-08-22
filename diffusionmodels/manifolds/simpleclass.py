"""
manifolds.structures.simpleclass
================================

A module that implements various simple manifolds

Classes
-------
Euclidean
    An arbitrary dimension manifold with flat geometry
"""


import torch

from .baseclass import Manifold


class Euclidean(Manifold):
    """
    A manifold with flat geometry

    Methods
    -------
    exp(X, dX)
        Increment a point X with a tangent vector dX

    log(X, Y)
        Calculate a tangent vector dX such that exp(X, dX) = Y
    """

    def __init__(self):
        pass


    def exp(self, X: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """
        Increment a point X with a tangent vector dX

        Parameters
        ----------
        X : torch.Tensor
            A point in the manifold

        dX : torch.Tensor
            A tangent vector in a tangent space on the manifold

        Returns
        -------
        torch.Tensor
            In Euclidean manifold, simply X + dX
        """
        return X + dX

    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Computes a tangent vector dX such that exp(X, dX) = Y

        Parameters
        ----------
        X : torch.Tensor
            The origin point in the manifold

        Y : torch.Tensor
            The destination point in the manifold

        Returns
        -------
        torch.Tensor
            In Euclidean manifold, simply Y - X
        """
        return Y - X
