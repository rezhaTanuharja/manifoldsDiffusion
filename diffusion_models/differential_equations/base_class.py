"""
diffusion_models.differential_equations.base_class
==================================================

This module defines the abstract base classes for differential_equations.

Classes
-------
stochastic_differential_equation
    Represents SDEs in the form of dX = drift(X, t) dt + diffusion(X, t) dW
"""


from abc import ABC, abstractmethod
import torch


class stochastic_differential_equation(ABC):
    """
    An abstract class of stochastic differential equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    Methods
    -------
    drift(X, t)
        Evaluate the drift tensor field at (X, t)

    diffusion(X, t)
        Evaluate the diffusion tensor field at (X, t)
    """


    def __init__(self) -> None:
        pass


    @abstractmethod
    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the drift tensor field at (X, t)

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            The temporal location to evaluate

        Returns
        -------
        torch.Tensor
            The drift tensor, drift(X, t)
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the diffusion tensor field at (X, t)

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            The temporal location to evaluate

        Returns
        -------
        torch.Tensor
            The diffusion term, diffusion(X, t) dW
        """
        raise NotImplementedError("Subclasses must implement this method")


class reversed_SDE(ABC):
    """
    An abstract class of a reversal of stochastic differential equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    Parameters
    ----------
    sde : stochastic_differential_equation
        The stochastic differential_equations to be reversed

    Methods
    -------
    drift(X, t)
        Evaluate the drift tensor field at (X, t)

    diffusion(X, t)
        Evaluate the diffusion tensor field at (X, t)
    """


    def __init__(self, sde: stochastic_differential_equation) -> None:
        self.sde = sde


    @abstractmethod
    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the drift tensor field at (X, t)

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            The temporal location to evaluate

        Returns
        -------
        torch.Tensor
            The drift tensor, drift(X, t)
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the diffusion tensor field at (X, t)

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            The temporal location to evaluate

        Returns
        -------
        torch.Tensor
            The diffusion term, diffusion(X, t) dW
        """
        raise NotImplementedError("Subclasses must implement this method")
