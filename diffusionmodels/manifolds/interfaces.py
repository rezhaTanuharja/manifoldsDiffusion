"""Provides the interface for all manifolds in this project.

Classes
-------
`Manifold`
A purely abstract class that serves as an interface of all manifolds
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch


class Manifold(ABC):
    """An abstract class that serves as an interface for all manifolds.

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    Properties
    ----------
    `dimension`
    The tensor shape of points on the manifold

    `tangent_dimension`
    The tensor shape of vectors on the tangent space
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Moves all tensor attributes to the given device.

        Parameters
        ----------
        `device: torch.device`
        A device object representing the target hardware

        Usage Example
        -------------
        ```python
        manifold.to(torch.device("cuda", 0))
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """Access the dimension of points on the manifold.

        Returns
        -------
        `Tuple[int, ...]`
        The tensor shape of points on the manifold, is always a tuple

        Usage Example
        -------------
        ```python
        assert manifold.dimension == (3, 3)
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def tangent_dimension(self) -> Tuple[int, ...]:
        """Access the dimension of vectors on the tangent space.

        Returns
        -------
        `Tuple[int, ...]`
        The tensor shape of vectors on the tangent space, is always a tuple

        Usage Example
        -------------
        ```python
        assert manifold.tangent_dimension == (3,)
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")


class RiemannianManifold(Manifold):
    """An abstract class that serves as an interface for all Riemannian manifolds.

    Methods (extends `Manifold`'s methods and properties)
    ---------------------------------------------------
    `exp(points, vectors)`
    Returns the results of incrementing the points by the vectors

    `log(starts, ends)`
    Returns vectors such that `exp(starts, vectors) = ends`
    """

    @abstractmethod
    def exp(self, points: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """Returns the results of incrementing the points by the vectors.

        Parameters
        ----------
        `points: torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        `vectors: torch.Tensor`
        Vectors on the tangent space: a tensor with shape `(..., *tangent_dimension)`

        Returns
        -------
        `torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        Usage Example
        -------------
        ```python
        end_points = manifold.exp(start_poins, tangent_vectors)
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def log(self, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """Returns the vectors such that `exp(starts, vectors) = ends`.

        Parameters
        ----------
        `starts: torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        `ends: torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        Returns
        -------
        `torch.Tensor`
        Vectors on the tangent space: a tensor with shape `(..., *tangent_dimension)`

        Usage Example
        -------------
        ```python
        vectors = manifold.log(starts, ends)
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def metric(self, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """Returns the distance between `starts` and `ends`.
        The argument names starts and ends have no meanings, just don't know what to use.

        Parameters
        ----------
        `starts: torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        `ends: torch.Tensor`
        Points on the manifold: a tensor with shape `(..., *dimension)`

        Returns
        -------
        `torch.Tensor`
        A tensor with shape `(...)`

        Usage Example
        -------------
        ```python
        distance = manifold.metric(starts, ends)
        ```
        """
        raise NotImplementedError("Subclasses must implement this method")
