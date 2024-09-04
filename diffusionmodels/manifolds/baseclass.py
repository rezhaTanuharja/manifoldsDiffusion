"""
manifolds.baseclass
===================

Provides the interface for all manifolds in this package

Classes
-------
Manifold
    An abstract class that serves as an interface of all manifolds
"""


import torch
from abc import ABC, abstractmethod
from typing import Tuple


# NOTE: All interfaces must be defined in the same style as this one

class Manifold(ABC):
    """
    An abstract class that serves as an interface of all manifolds

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    `dimension()`
        Returns the tensor shape of each point in the manifold

    `tangent_dimension()`
        Returns the tensor shape of each vector in the manifold tangent space

    `exp(X, dX)`
        Increment a point X with a tangent vector dX

    `log(X, Y)`
        Calculate a tangent vector dX such that exp(X, dX) = Y
    """


    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def to(self, device: torch.device) -> None:
        """
        Move any tensor attributes to device

        Parameters
        ----------
        `device: torch.device`
            A device object from PyTorch
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
            The tensor shape of each point in the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def tangent_dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
            The tensor shape of each vector in the manifold tangent space
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def exp(self, X: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """
        Increment a point X with a tangent vector dX.

        Parameters
        ----------
        `X : torch.Tensor`
            A point in the manifold

        `dX : torch.Tensor`
            A tangent vector in a tangent space on the manifold

        Returns
        -------
        `torch.Tensor`
            A point in the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Computes a tangent vector dX such that exp(X, dX) = Y

        Parameters
        ----------
        `X : torch.Tensor`
            The origin point in the manifold

        `Y : torch.Tensor`
            The destination point in the manifold

        Returns
        -------
        `torch.Tensor`
            A tangent vector on the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")
