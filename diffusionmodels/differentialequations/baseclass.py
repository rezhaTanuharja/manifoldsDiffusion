"""
differentialequations.baseclass
===============================

Provides the interface of stochastic differential equations

Classes
-------
StochasticDifferentialEquation
    SDEs in the form of dX = drift(X, t) dt + diffusion(X, t) dW
"""


from abc import ABC, abstractmethod
import torch

from ..manifolds import Manifold


class StochasticDifferentialEquation(ABC):
    """
    An abstract class of stochastic differential equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    Methods
    -------
    manifold()
        Provides access to the manifold the SDE lives in

    drift(X, t)
        Evaluate the drift tensor field at (X, t)

    diffusion(X, t)
        Evaluate the diffusion tensor field at (X, t)
    """


    @abstractmethod
    def manifold(self) -> Manifold:
        """
        Returns
        -------
        Manifold
            A manifold the SDE lives in
        """
        raise NotImplementedError("Subclasses must implement this method")



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


class InitialValueProblems:
    """
    A list of initial-value problems, i.e., a list of dictionaries with keys:
        - 'initial_condition'
        - 'stochastic_de'

    Behaves like a list of dictionaries

    Methods
    -------
    append(initial_condition, stochastic_de)
        Check if initial condition and SDE are compatible and add to the list

    Private Attributes
    ------------------
    _elements
        A list of dictionaries with keys as described above
    """


    def __init__(self):
        self._elements = []


    def __iter__(self):
        return iter(self._elements)


    def append(
        self,
        initial_condition: torch.Tensor,
        stochastic_de: StochasticDifferentialEquation
    ):
        """
        Check if initial condition and SDE are compatible and add to the list

        Parameters
        ----------
        initial_condition : torch.Tensor
            A point in a manifold that serves as X(0)

        stochastic_de : StochasticDifferentialEquation
            A stochastic differential equation
        """

        if initial_condition.dim() < 3:
            raise ValueError("Initial condition must have more than 2 dimensions")

        if initial_condition.shape[2:] != stochastic_de.manifold().dimension():
            raise ValueError("Initial condition and SDE don't live in the same manifold")

        self._elements.append(
            {
                'initial_condition': initial_condition,
                'stochastic_de': stochastic_de
            }
        )

