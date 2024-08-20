"""
diffusionmodels.differentialequations.reversedclass
===================================================

This module implements classes of reversed differentialequations

Classes
-------
CorrectedNegative
    Represent reversal by multiplying the drift with -1 and applying correction
"""


import torch

from .baseclass import StochasticDifferentialEquation, ReversedSDE
from ..scorefunctions import DirectionCalculator


class CorrectedNegative(ReversedSDE):
    """
    This class reverses the processes described by differential_equations in the form of

        dX = drift(X, t) dt + diffusion(X, t) dW

    by inverting the drift direction and applying a correction:

        dX = [-drift(X, t) + correction(X, t)] dt + diffusion(X, t) dW

    The correction is proportional to the square of diffusion

    Parameters
    ----------
    sde : StochasticDifferentialEquation
        The stochastic differential_equations to be reversed

    drift_corrector : DirectionCalculator
        A function that takes X as input and output the drift correction
    """


    def __init__(
        self,
        sde: StochasticDifferentialEquation,
        drift_corrector: DirectionCalculator
    ):
        super().__init__(sde)
        self.drift_corrector = drift_corrector



    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the reversed drift tensor of the SDE

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            The current time

        Returns
        -------
        torch.Tensor
            drift = -sde.drift + diffusion * diffusion * correction
        """

        return (
            -self.sde.drift(X, t) + (
                self.sde.diffusion(X, t) ** 2
            ) * self.drift_corrector.get_direction(X, t)
        )


    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the diffusion tensor of the reversed SDE

        Parameters
        ----------
        X : torch.Tensor
            The current spatial location, i.e., X(t)

        t : float
            The current time

        Returns
        -------
        torch.Tensor
            diffusion = sde.diffusion(X, t)
        """

        return self.sde.diffusion(X, t)
