"""
distributions.interfaces
========================

Provides the interface for all distributions in this package

Classes
-------
Distribution        : The interface for all distributions in this package
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class StochasticProcess(ABC):
    """
    The interface for all stochastic processes in the Eulerian module

    Methods
    -------
    `dimension()`
        Returns the dimension of the random variable

    `to(device)`
        Send process to device

    `at(time)`
        Access the process at the given time

    `density(points)`
        Compute the probability density function at the given points

    `sample(num_samples)`
        Generate a number of random samples from the process
    """


    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        `Tuple[int, ...]`
            The dimension of the random variable, not counting the time
        """
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
    def at(self, time: torch.Tensor) -> None:
        """
        Fixes the time so the stochastic process becomes a 'static' random variable

        Parameters
        ----------
        `time: torch.Tensor`
            The time tensor to access the process
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density function at the given points

        Parameters
        ----------
        `points: torch.Tensor`
            The points where PDF will be computed

        Returns
        -------
        `torch.Tensor`
            The value of the PDF at the given points
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def sample(self, num_samples: int, *args, **kwargs) -> torch.Tensor:
        """
        Generate a number of random samples from the distribution

        Parameters
        ----------
        `num_samples: int`
            The number of random samples to generate

        Returns
        -------
        `torch.Tensor`
            A tensor of random samples, the first index is the sample index
        """
        raise NotImplementedError("Subclasses must implement this method")