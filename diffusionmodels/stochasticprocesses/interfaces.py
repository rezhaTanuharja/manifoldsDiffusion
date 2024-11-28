"""
Provides the interfaces for all stochastic processes in this project.

Classes
-------
`DensityFunction`
A purely abstract class that serves as an interface of all density functions

`StochasticProcess`
A purely abstract class that serves as an interface of all stochastic processes
"""


from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import jax
import jax.numpy as jnp


class DensityFunction(ABC):
    """
    An abstract class that serves as an interface of all density functions

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `__call__(points, times)`
    Evaluate the density value at the given points and times

    `gradient(points, times)`
    Evalute the gradient of density at the given points and times
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
    def __call__(
        self, points: List[jnp.ndarray], times: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Evaluate the density value at the given points and times

        Parameters
        ----------
        `points: List[jnp.ndarray]`
        Array with shape `(..., time_index, num_points, *dimension)`.
        The dimension `time_index` must be broadcastable to `num_times`

        `times: Optional[jnp.ndarray]`
        Array with shape `(..., num_times)` or `None`

        Returns
        -------
        `jnp.ndarray`
        Array with shape `(..., num_times, num_points)`
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def gradient(
        self, points: List[jnp.ndarray], times: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Evaluate the gradient of density at the given points and times

        Parameters
        ----------
        `points: List[jnp.ndarray]`
        Array with shape `(..., time_index, num_points, *dimension)`.
        The dimension `time_index` must be broadcastable to `num_times`

        `times: Optional[jnp.ndarray]`
        Array with shape `(..., num_times)` or `None`

        Returns
        -------
        `jnp.ndarray`
        Array with shape `(..., num_times, num_points, *dimension)`
        """
        raise NotImplementedError("Subclasses must implement this method")


class StochasticProcess(ABC):
    """
    An abstract class that serves as an interface of all stochastic processes

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `dimension()`
    Returns the shape of points on the manifold

    `density()`
    Provides access to the underlying density function

    `sample(num_samples, times)`
    Generate a number of random samples from the process at the given times
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
        The shape of each realization point of the process, always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def density(self) -> DensityFunction:
        """
        Provides access to the underlying density function

        Returns
        -------
        `DensityFunction | Callable[[List[jnp.ndarray]], jnp.ndarray]`
        A callable object with the following call args:
            `points: List[jnp.ndarray]`
            `times : jnp.ndarray`
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def sample(
        self, num_samples: int, times: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Generate random samples from the stochastic process

        Parameters
        ----------
        `num_samples: int`
        The number of samples to generate

        `times: Optional[jnp.ndarray]`
        Array with shape `(..., num_times)` or `None`

        Returns
        -------
        `jnp.ndarray`
        Array with shape `(..., num_times, num_samples)` or `(..., num_samples)`
        """
        raise NotImplementedError("Subclasses must implement this method")
