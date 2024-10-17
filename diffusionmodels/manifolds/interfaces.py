"""
manifolds.interfaces
====================

Provides the interface for all manifolds in this package

Classes
-------
Manifold
    An abstract class that serves as an interface of all manifolds
"""


import torch
from abc import ABC, abstractmethod
from typing import Tuple


class Manifold(ABC):
    """
    An abstract class that serves as an interface of all manifolds

    Methods
    -------
    `to(device)`
        Moves any tensor attribute to device

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
    def exp(self, points: torch.Tensor, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Increment points with tangent vectors.
        In Euclidean geometry, simply returns `points + tangent_vector`

        Parameters
        ----------
        `points : torch.Tensor`
            The original points in the manifold

        `tangent_vector: torch.Tensor`
            Vectors that live in the manifold tangent space

        Returns
        -------
        `torch.Tensor`
            New points in the manifold
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def log(self, origins: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
        """
        Computes vectors such that `exp(origins, vectors) = destinations`

        Parameters
        ----------
        `origins : torch.Tensor`
            The points of origin in the manifold

        `destinations : torch.Tensor`
            The points of destination in the manifold

        Returns
        -------
        `torch.Tensor`
            Vectors that increments origins to their respective destinations
        """
        raise NotImplementedError("Subclasses must implement this method")
