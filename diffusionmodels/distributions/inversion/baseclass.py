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
    `solve(values, function)`
        Find the points where function evaluates to the given values
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def solve(
        self,
        values: torch.Tensor,
        function: CumulativeDistributionFunction
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")
