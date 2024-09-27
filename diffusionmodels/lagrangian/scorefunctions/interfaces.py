"""
lagrangian.scorefunctions.interfaces
====================================

This module defines the abstract base classes for scorefunctions

Classes
-------
DirectionCalculator
    An abstract class that computes a direction using the current position
"""


from abc import ABC, abstractmethod
import torch


class Direction(ABC):
    """
    An abstract class that computes a direction from an origin to a destination

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    get_direction(origin, destination, scale)
        Compute direction from the origin to destination multiplied by the scale
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
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
    def get_direction(
        self,
        origin: torch.Tensor, destination: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Compute the direction, dX, to update X(t)

        Parameters
        ----------
        X : torch.Tensor
            The current spatial location, i.e., X(t)

        t : float
            The current time

        Returns
        -------
        torch.Tensor
            The direction of update, dX
        """
        raise NotImplementedError("Subclasses must implement this method")
