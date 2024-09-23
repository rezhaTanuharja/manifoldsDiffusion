"""
distributions.baseclass
=======================

Provides the interface for all distributions in this package

Classes
-------
Distribution        : The interface for all distributions in this package
"""

from abc import ABC, abstractmethod
import torch


class Distribution(ABC):
    """
    The interface for all distributions in this package

    Methods
    -------
    `to(device)`
        Send all tensor attributes to device

    `sample(num_samples)`
        Generate a number of random samples from the distribution
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def to(self, device: torch.device) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def at(self, time: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
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
