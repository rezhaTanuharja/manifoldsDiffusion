"""
scorefunctions.baseclass
========================

This module defines the abstract base classes for scorefunctions

Classes
-------
DirectionCalculator
    An abstract class that computes a direction using the current position

RelativeDirectionCalculator
    An abstract class that computes a direction using the current position and a reference positions
"""


from abc import ABC, abstractmethod
import torch

from ..manifolds import Manifold


class RelativeDirectionCalculator(ABC):
    """
    An abstract class that computes a direction using the current position and a reference position

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    get_direction(X, X_ref, time, t_ref)
        Compute the direction to update X(t)
    """


    @abstractmethod
    def __init__(self, manifold: Manifold) -> None:
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
