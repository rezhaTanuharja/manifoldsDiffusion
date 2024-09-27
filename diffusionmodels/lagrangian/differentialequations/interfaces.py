"""
lagrangian.differentialequations.interfaces
===========================================

Provides the interface of stochastic differential equations

Classes
-------
StochasticDifferentialEquation
    SDEs in the form of dX = drift(X, t) dt + diffusion(X, t) dW
"""

from abc import ABC, abstractmethod
import torch

from ...manifolds import Manifold


class StochasticDifferentialEquation(ABC):
    """
    An abstract class of stochastic differential equations in the form of

        `dX = drift(X, t) dt + diffusion(X, t) dW`

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    `manifold()`
        Provides access to the manifold the SDE lives in

    `drift(X, t)`
        Evaluate the drift tensor field at (X, t)

    `diffusion(X, t)`
        Evaluate the diffusion tensor field at (X, t)
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
    def manifold(self) -> Manifold:
        """
        Returns
        -------
        `Manifold`
            A manifold the SDE lives in
        """
        raise NotImplementedError("Subclasses must implement this method")



    @abstractmethod
    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the drift tensor field at (X, t)

        Parameters
        ----------
        `X : torch.Tensor`
            The spatial location to evaluate

        `t : float`
            The temporal location to evaluate

        Returns
        -------
        `torch.Tensor`
            The drift tensor, drift(X, t)
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the diffusion tensor field at (X, t)

        Parameters
        ----------
        `X : torch.Tensor`
            The spatial location to evaluate

        `t : float`
            The temporal location to evaluate

        Returns
        -------
        `torch.Tensor`
            The diffusion term, diffusion(X, t) dW
        """
        raise NotImplementedError("Subclasses must implement this method")
