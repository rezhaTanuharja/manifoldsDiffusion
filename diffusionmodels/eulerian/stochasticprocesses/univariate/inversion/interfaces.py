"""
eulerian.stochasticprocesses.univariate.inversion.interfaces
============================================================

Provides the interface for all inversion method in the Eulerian module

Classes
-------
InversionMethod
    An abstract class that serves as an interface of all inversion methods
"""


from abc import ABC, abstractmethod
from typing import Callable, Dict
import torch


class InversionMethod(ABC):
    """
    An abstract class that serves as an interface of all inversion methods

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
        function: Callable[[torch.Tensor], torch.Tensor],
        search_range: Dict[str, float]
    ) -> torch.Tensor:
        """
        Invert a given function to find the preimage of the given values

        Parameters
        ----------
        `values: torch.Tensor`
            The function values

        `function: Callable[[torch.Tensor], torch.Tensor]`
            An object that takes a tensor as an input and output the same-sized tensor

        `search_range: Dict[str, float]`
            A dictionary with keys 'lower_bound' and 'upper_bound'

        Returns
        -------
        `torch.Tensor`
            The points where the function evaluates to the given values
        """
        raise NotImplementedError("Subclasses must implement this method")
