"""
score_functions.base_class
==========================

This module defines the abstract base classes for score_functions

Classes
-------
direction_calculator
    An abstract class that computes a direction using the current position

relative_direction_calculator
    An abstract class that computes a direction using the current position and a reference positions
"""


from abc import ABC, abstractmethod
import torch


class direction_calculator(ABC):
    """
    An abstract class that computes a direction using the current position

    Methods
    -------
    get_direction(X, t)
        Compute the direction to update X(t)
    """


    def __init__(self):
        pass



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


class relative_direction_calculator(direction_calculator):
    """
    An abstract class that computes a direction using the current position and a reference position

    Methods
    -------
    get_direction(X, X_ref, t, t_ref)
        Compute the direction to update X(t)
    """


    def __init__(self, X_ref, t_ref):
        super().__init__()
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
