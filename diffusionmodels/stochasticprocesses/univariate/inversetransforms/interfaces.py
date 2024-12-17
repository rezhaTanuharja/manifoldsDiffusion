"""
Provides the interfaces for all univariate processes in this project.

Classes
-------
`CumulativeDistributionFunction`
A purely abstract class that serves as an interface of all CDFs

`RootFinder`
A purely abstract class that serves as an interface of all root-finder
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import torch

from ...interfaces import DensityFunction


class CumulativeDistributionFunction(ABC):
    """
    An abstract class that serves as an interface of all CDFs

    Methods
    -------
    `to(device)`
    Moves all tensor attributes to the given device

    `__call__(points, times)`
    Evaluate the CDF value at the given points and times

    Properties
    ----------
    `gradient`
    An instance of `DensityFunction`

    `support`
    A dict with key 'lower' and 'upper'
    Indicate the interval in which CDF may have non-zero values
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

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
    def __call__(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the CDF value at the given points and times

        Parameters
        ----------
        `points: torch.Tensor`
        Tensor with shape `(..., num_times, num_points, *dimension)`.

        `times: torch.Tensor`
        Tensor with shape `(..., num_times)`

        Returns
        -------
        `torch.Tensor`
        Tensor with shape `(..., num_times, num_points)`
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
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

    @property
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
        function: Callable[[torch.Tensor], torch.Tensor],
        target_values: torch.Tensor,
        interval: Tuple[float, float],
    ) -> torch.Tensor:
        """
        Find the solution of `function(points) = target_values` inside the `interval`

        Parameters
        ----------
        `function: Callable[[torch.Tensor], torch.Tensor]`
        A mapping `R ** (*input_shape) -> R ** (*output_shape)`

        `target_values: torch.Tensor`
        An array with shape `(*output_shape)`

        `interval: Tuple[float, float]`
        A tuple containing the lower and upper bound of the interval

        Returns
        -------
        `torch.Tensor`
        An array with shape `(*input_shape)`
        """
        raise NotImplementedError("Subclasses must implement this method")