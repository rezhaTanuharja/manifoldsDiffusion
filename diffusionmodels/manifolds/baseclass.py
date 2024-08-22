"""
diffusionmodels.manifolds.baseclass
===================================

A module that defines abstract classes of manifolds
"""


import torch
from abc import ABC, abstractmethod


class Manifold(ABC):
    """
    An abstract class that provides manifold structures

    Methods
    -------
    exp(X, dX)
        Increment a point X with a tangent vector dX

    log(X, Y)
        Calculate a tangent vector dX such that exp(X, dX) = Y
    """

    def __init__(self):
        pass


    @abstractmethod
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
            A point in the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
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
            A tangent vector on the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")
