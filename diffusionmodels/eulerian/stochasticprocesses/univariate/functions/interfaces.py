"""
distributions.functions.interfaces
==================================

Provides the interface for all PDFs and CDFs in this package

Classes
-------
DistributionFunction
    An abstract class that serves as an interface of all distribution functions
"""


from abc import ABC, abstractmethod
from typing import Dict

import torch


class CumulativeDistributionFunction(ABC):
    """
    An abstract callable class that serves as an interface of all CDFs

    Methods
    -------
    `to(device)`
        Move any tensor attribute to device

    `at(time)`
        Access the CDF at the given time

    `gradient(points)`
        Returns the PDF values at the given points

    `hessian(points)`
        Returns the gradient of PDF values at the given points

    `support()`
        Return a dictionary containing lower and upper bound of the function's support
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
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
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
    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `points: torch.Tensor`
            The points to evaluate the CDF gradient

        Returns
        -------
        `torch.Tensor`
            The gradient, i.e., probability density value at the given points
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def hessian(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `points: torch.Tensor`
            The points to evaluate the CDF gradient

        Returns
        -------
        `torch.Tensor`
            The gradient of the probability density value at the given points
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def support(self) -> Dict[str, float]:
        """
        Support is the range in which the CDF may be nonzero

        Returns
        -------
        `Dict[str, float]`
            A dictionary with keys 'lower_bound' and 'upper_bound'
        """
        raise NotImplementedError("Subclasses must implement this method")
