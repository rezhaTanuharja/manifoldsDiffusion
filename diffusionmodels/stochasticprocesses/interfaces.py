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
from typing import Tuple

import torch


class DensityFunction(ABC):
    """
    An abstract class that serves as an interface of all density functions

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `at(time)`
    Set the time to access the density function

    `__call__(points)`
    Evaluate the density value at the given points

    `gradient(points)`
    Evalute the gradient of density at the given points

    Properties
    ----------
    `dimension`
    The tensor shape of each realization point
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

    @abstractmethod
    def at(self, time: torch.Tensor):
        """
        Set the internal time of the density function

        Parameters
        ----------
        `time: torch.Tensor`
        Tensor with shape `(..., num_times)`

        Returns
        -------
        `DensityFunction`
        The same density function with internal time set to the given tensor
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
        The shape of each realization point, always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the density value at the given points and times

        Parameters
        ----------
        `points: torch.Tensor`
        Tensor with shape `(..., time_index, num_points, *dimension)`.
        The dimension `time_index` must be broadcastable to `num_times`
        If `*dimension == 1` then it is omitted from the shape

        Returns
        -------
        `torch.Tensor`
        Tensor with shape `(..., num_times, num_points)`
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gradient of density at the given points and times

        Parameters
        ----------
        `points: torch.Tensor`
        Tensor with shape `(..., time_index, num_points, *dimension)`.
        The dimension `time_index` must be broadcastable to `num_times`
        If `*dimension == 1` then it is omitted from the shape

        Returns
        -------
        `torch.Tensor`
        Tensor with shape `(..., num_times, num_points, *dimension)`
        If `*dimension == 1` then it is omitted from the shape
        """
        raise NotImplementedError("Subclasses must implement this method")


class StochasticProcess(ABC):
    """
    An abstract class that serves as an interface of all stochastic processes

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `at(time)`
    Set the time to access the stochastic process

    `sample(num_samples, times)`
    Generate a number of random samples from the process at the given times

    Properties
    ----------
    `dimension`
    The tensor shape of each realization point

    `density`
    The underlying density function of the process
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
        A device object from Jax representing the target hardware
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def at(self, time: torch.Tensor):
        """
        Set the internal time of the stochastic process

        Parameters
        ----------
        `time: torch.Tensor`
        Tensor with shape `(..., num_times)`

        Returns
        -------
        `DensityFunction`
        The same stochastic process with internal time set to the given tensor
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
        The shape of each realization point of the process, always a tuple
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def density(self) -> DensityFunction:
        """
        Provides access to the underlying density function

        Returns
        -------
        `DensityFunction`
        A callable object that maps `torch.Tensor` to another `torch.Tensor` with the same shape
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate random samples from the stochastic process

        Parameters
        ----------
        `num_samples: int`
        The number of samples to generate

        Returns
        -------
        `torch.Tensor`
        Tensor with shape `(..., num_times, num_samples)`
        """
        raise NotImplementedError("Subclasses must implement this method")
