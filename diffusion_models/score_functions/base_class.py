"""
score_functions.base_class
==========================

This module defines the abstract base classes for score_functions

Classes
-------
explicit
    Represents an explicit time integration
"""


from abc import ABC, abstractmethod
import torch


class direction_calculator(ABC):
    """
    A generalization of the Stein score function, which calculate the direction to maximize the increase in log-probability. This abstract class compute a direction that achieves certain criterias specified by the child classes.

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
            The spatial location to evaluate

        t : float
            The temporal location to evaluate

        Returns
        -------
        torch.Tensor
            The direction of update, dX
        """
        raise NotImplementedError("Subclasses must implement this method")
