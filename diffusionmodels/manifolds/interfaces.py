"""
manifolds.interfaces
====================

Provides the interface for all manifolds in this project

Classes
-------
Manifold
    A purely abstract class that serves as an interface of all manifolds
"""


from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp


class Manifold(ABC):
    """
    An abstract class that serves as an interface of all manifolds

    Methods
    -------
    `to(device)`
        Moves all tensor attributes to the given device

    `dimension()`
        Returns the shape of points on the manifold

    `tangent_dimension()`
        Returns the shape of vectors on the tangent space

    `exp(points, vectors)`
        Returns the results of incrementing the points by the vectors

    `log(starts, ends)`
        Returns the vectors such that `exp(starts, vectors) = ends`
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def to(self, device: jax.Device) -> None:
        """
        Moves all tensor attributes to the given device

        Parameters
        ----------
        `device: jax.Device`
            A device object from Jax representing the target hardware
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
            The shape of points on the manifold, is always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def tangent_dimension() -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
            The shape of vectors on the tangent space, is always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def exp(self, points: jnp.ndarray, vectors: jnp.ndarray) -> jnp.ndarray:
        """
        Returns the results of incrementing the points by the vectors

        Parameters
        ----------
        `points: jax.numpy.ndarray`
            Points on the manifold with shape `(..., *dimension)`

        `vectors: jax.numpy.ndarray`
            Vectors on the tangent space with shape `(..., *tangent_dimension)`

        Returns
        -------
        `jax.numpy.ndarray`
            Points on the manifold with shape `(..., *dimension)`
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def log(self, starts: jnp.ndarray, ends: jnp.ndarray) -> jnp.ndarray:
        """
        Returns the vectors such that `exp(starts, vectors) = ends`

        Parameters
        ----------
        `starts: jax.numpy.ndarray`
            Points on the manifold with shape `(..., *dimension)`

        `ends: jax.numpy.ndarray`
            Points on the manifold with shape `(..., *dimension)`

        Returns
        -------
        `jax.numpy.ndarray`
            Vectors on the tangent space with shape `(..., *tangent_dimension)`
        """
        raise NotImplementedError("Subclasses must implement this method")
