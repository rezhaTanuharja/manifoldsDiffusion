"""
diffusionmodels.scorefunctions.baseclass
===========================================

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


class DirectionCalculator(ABC):
    """
    An abstract class that computes a direction using the current position

    Methods
    -------
    get_direction(X, t)
        Compute the direction to update X(t)
    """


    def __init__(self, manifold: Manifold) -> None:
        self.manifold = manifold



    @abstractmethod
    def get_direction(
        self,
        X: torch.Tensor,
        t: float
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


class RelativeDirectionCalculator(DirectionCalculator):
    """
    An abstract class that computes a direction using the current position and a reference position

    Methods
    -------
    get_direction(X, X_ref, time, t_ref)
        Compute the direction to update X(t)
    """


    def __init__(
        self,
        manifold: Manifold,
        X_ref: torch.Tensor,
        t_ref: torch.Tensor
    ):
        super().__init__(manifold)
        self.X_ref = X_ref
        self.t_ref = t_ref



    @abstractmethod
    def get_direction(
        self,
        X: torch.Tensor,
        t: float,
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
