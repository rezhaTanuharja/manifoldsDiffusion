"""
diffusionmodels.differentialequations.baseclass
===============================================

This module defines the abstract base classes for differentialequations.

Classes
-------
StochasticDifferentialEquation
    Represents SDEs in the form of dX = drift(X, t) dt + diffusion(X, t) dW

ReversedSDE
    Represents the reversal of SDEs in the form of dX = drift(X, t) dt + diffusion(X, t) dW
"""


from abc import ABC, abstractmethod
import torch

from ..manifolds import Manifold


class StochasticDifferentialEquation(ABC):
    """
    An abstract class of stochastic differential equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    Attributes
    ----------
    manifold : Manifold
        Provides the manifold structures where the SDE lives

    Methods
    -------
    drift(X, t)
        Evaluate the drift tensor field at (X, t)

    diffusion(X, t)
        Evaluate the diffusion tensor field at (X, t)
    """


    def __init__(self, manifold: Manifold) -> None:
        self.manifold = manifold


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


class ReversedSDE(StochasticDifferentialEquation):
    """
    An abstract class of a reversal of stochastic differential equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    Attributes
    ----------
    manifold : Manifold
        Inherited from sde

    sde : StochasticDifferentialEquation
        The stochastic differential equations to be reversed

    Methods
    -------
    drift(X, t)
        Evaluate the drift tensor field at (X, t)

    diffusion(X, t)
        Evaluate the diffusion tensor field at (X, t)
    """


    def __init__(self, sde: StochasticDifferentialEquation) -> None:
        super().__init__(sde.manifold)
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
