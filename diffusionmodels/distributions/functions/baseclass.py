"""
distributions.functions.baseclass
=================================

Provides the interface for all PDFs and CDFs in this package

Classes
-------
CumulativeDistributionFunction
    An abstract class that serves as an interface of all CDFs
"""


from abc import ABC, abstractmethod
from typing import Dict

import torch


class CumulativeDistributionFunction(ABC):
    """
    An abstract class that serves as an interface of all CDFs

    Methods
    -------
    `to(device)`
        Move any tensor attribute to device

    `at(time)`
        Access the CDF at the given time

    `evaluate(points)`
        Returns the CDF values at the given points

    `boundaries()`
        Return a dictionary containing lower and upper bound of the CDF domain
    """


    @abstractmethod
    def __init__(self, *args, **kwargs):
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
    def at(self, time: float):
        """
        Access the CDF at the given time, does nothing for a time-invariant distribution

        Parameters
        ----------
        `time: float`
            The time to evaluate the CDF

        Returns
        -------
        `self`
            The same instance of distribution after it is moved to the given time
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
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
    def boundaries(self) -> Dict[str, float]:
        """
        Returns
        -------
        `Dict[str, float]`
            A dictionary with keys 'lower_bound' and 'upper_bound'
        """
        raise NotImplementedError("Subclasses must implement this method")
