"""
Provides the interfaces for all univariate processes in this project.

Classes
-------
`CumulativeDistributionFunction`
A purely abstract class that serves as an interface of all CDFs

`RootFinder`
A purely abstract class that serves as an interface of all root-finder
"""


from .. import DensityFunction

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable

import jax
import jax.numpy as jnp


class CumulativeDistributionFunction(ABC):
    """
    An abstract class that serves as an interface of all CDFs

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `__call__(points, times)`
    Evaluate the CDF value at the given points and times

    `gradient(points, times)`
    Evalute the PDF value at the given points and times

    `support()`
    Provides the range in which CDF may have non-zero value
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass


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
        self, points: jnp.ndarray, times: jnp.ndarray = jnp.array([0.0,])
    ) -> jnp.ndarray:
        """
        Evaluate the CDF value at the given points and times

        Parameters
        ----------
        `points: jnp.ndarray`
        Array with shape `(..., time_index, num_points, *dimension)`.
        The dimension `time_index` must be broadcastable to `num_times`

        `times: jnp.ndarray = jnp.array([0.0,])`
        Array with shape `(..., num_times)`

        Returns
        -------
        `jnp.ndarray`
        Array with shape `(..., num_times, num_points)`
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def gradient(self) -> DensityFunction:
        """
        Evaluate the PDF value at the given points and times

        Returns
        -------
        `DensityFunction`
        Provides access to the underlying density function
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def support(self) -> Dict[str, float]:
        """
        Provides the range in which CDF may have non-zero value

        Returns
        -------
        `Dict[str, float]`
        A dictionary with the keys `lower_bound` and `upper_bound`
        """
        raise NotImplementedError("Subclasses must implement this method")


class RootFinder(ABC):
    """
    A purely abstract class that serves as an interface of all root-finder

    Methods
    -------
    `solve(function, target_values, interval)`
    Find the solution of `function(points) = target_values` inside the `interval`
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def solve(

        self,
        function: Callable[[jnp.ndarray], jnp.ndarray],
        target_values: jnp.ndarray,
        interval: Tuple[float, float]

    ) -> jnp.ndarray:
        """
        Find the solution of `function(points) = target_values` inside the `interval`

        Parameters
        ----------
        `function: Callable[[jnp.ndarray], jnp.ndarray]`
        A mapping `R ** (*input_shape) -> R ** (*output_shape)`

        `target_values: jnp.ndarray`
        An array with shape `(*output_shape)`

        `interval: Tuple[float, float]`
        A tuple containing the lower and upper bound of the interval

        Returns
        -------
        `jnp.ndarray`
        An array with shape `(*input_shape)`
        """
        raise NotImplementedError("Subclasses must implement this method")
