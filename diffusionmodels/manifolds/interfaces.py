"""
Provides the interface for all manifolds in this project.

Classes
-------
`Manifold`
A purely abstract class that serves as an interface of all manifolds
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Manifold(ABC):
    """
    An abstract class that serves as an interface of all manifolds

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `exp(points, vectors)`
    Returns the results of incrementing the points by the vectors

    `log(starts, ends)`
    Returns the vectors such that `exp(starts, vectors) = ends`

    Properties
    ----------
    `dimension`
    The tensor shape of points on the manifold

    `tangent_dimension`
    The tensor shape of vectors on the tangent space
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """
        Moves all tensor attributes to the given device

        Parameters
        ----------
        `device: torch.device`
        A device object representing the target hardware
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
        The tensor shape of points on the manifold, is always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def tangent_dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
        The tensor shape of vectors on the tangent space, is always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def exp(self, points: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """
        Returns the results of incrementing the points by the vectors

        Parameters
        ----------
        `points: torch.Tensor`
        Points on the manifold with shape `(..., *dimension)`

        `vectors: torch.Tensor`
        Vectors on the tangent space with shape `(..., *tangent_dimension)`

        Returns
        -------
        `torch.Tensor`
        Points on the manifold with shape `(..., *dimension)`
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def log(self, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """
        Returns the vectors such that `exp(starts, vectors) = ends`

        Parameters
        ----------
        `starts: torch.Tensor`
        Points on the manifold with shape `(..., *dimension)`

        `ends: torch.Tensor`
        Points on the manifold with shape `(..., *dimension)`

        Returns
        -------
        `torch.Tensor`
        Vectors on the tangent space with shape `(..., *tangent_dimension)`
        """
        raise NotImplementedError("Subclasses must implement this method")
