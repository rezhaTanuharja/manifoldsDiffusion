"""
distributions.functions.baseclass
=================================

Provides the interface for all PDFs and CDFs in this package

Classes
-------
DistributionFunction
    An abstract class that serves as an interface of all distribution functions
"""


from abc import ABC, abstractmethod
from typing import Dict

import torch


class DistributionFunction(ABC):
    """
    An abstract class that serves as an interface of all CDFs

    Methods
    -------
    `to(device)`
        Move any tensor attribute to device

    `at(time)`
        Access the CDF at the given time

    `cumulative(points)`
        Returns the CDF values at the given points

    `density(points)`
        Returns the PDF values at the given points

    `support()`
        Return a dictionary containing lower and upper bound of the function's support
    """


    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def cumulative(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `points: torch.Tensor`
            The points to evaluate the CDF

        Returns
        -------
        `torch.Tensor`
            The probability that a random sample is lower than or equal to point
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `points: torch.Tensor`
            The points to evaluate the CDF

        Returns
        -------
        `torch.Tensor`
            The probability density value at the given points
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
    def at(self, time: torch.Tensor):
        """
        Access the CDF at the given time, does nothing for a time-invariant distribution

        Parameters
        ----------
        `time: torch.Tensor`
            The time to evaluate the CDF

        Returns
        -------
        `self`
            The same instance of distribution after it is moved to the given time
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def support(self) -> Dict[str, float]:
        """
        Support is the range in which the functions may be nonzero

        Returns
        -------
        `Dict[str, float]`
            A dictionary with keys 'lower_bound' and 'upper_bound'
        """
        raise NotImplementedError("Subclasses must implement this method")
