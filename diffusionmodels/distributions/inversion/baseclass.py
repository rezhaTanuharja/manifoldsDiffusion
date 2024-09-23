"""
distributions.inversion.baseclass
=================================

Provides the interface for all inversion method in this package

Classes
-------
InversionMethod
    An abstract class that serves as an interface of all inversion method
"""


from abc import ABC, abstractmethod
import torch

from ..functions import CumulativeDistributionFunction


class InversionMethod(ABC):
    """
    An abstract class that serves as an interface of all inversion method

    Methods
    -------
    `to(device)`
        Move any tensor attribute to device

    `solve(values, function)`
        Find the points where function evaluates to the given values
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
    def solve(
        self,
        values: torch.Tensor,
        function: CumulativeDistributionFunction
    ) -> torch.Tensor:
        """
        Invert a given function to find the preimage of the given values

        Parameters
        ----------
        `values: torch.Tensor`
            The function values

        `function: CumulativeDistributionFunction`
            The function to be inverted

        Returns
        -------
        `torch.Tensor`
            The points where the function evaluates to the given values
        """
        raise NotImplementedError("Subclasses must implement this method")
