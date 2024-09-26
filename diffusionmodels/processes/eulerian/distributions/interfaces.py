"""
distributions.interfaces
========================

Provides the interface for all distributions in this package

Classes
-------
Distribution        : The interface for all distributions in this package
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import torch


class Distribution(ABC):
    """
    The interface for all distributions in this package

    Methods
    -------
    `to(device)`
        Send all tensor attributes to device

    `at(time)`
        Access distribution at the given time

    `sample(num_samples)`
        Generate a number of random samples from the distribution
    """


    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def dimension(self) -> Tuple[int, ...]:
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
        Distribution is a temporal and spatial function.
        This function fixes the time so it becomes a spatial-only function.

        Parameters
        ----------
        `time: torch.Tensor`
            The time tensor to access the distribution
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def density_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
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
