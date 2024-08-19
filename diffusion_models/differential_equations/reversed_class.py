"""
diffusion_models.differential_equations.reversed_class
======================================================

This module implements classes of reversed differential_equations

Classes
-------
corrected_negative
    Represent reversal by multiplying the drift with -1 and applying correction
"""


import torch

from .base_class import stochastic_differential_equation, reversed_SDE
from ..score_functions import direction_calculator


class corrected_negative(reversed_SDE):
    """
    This class reverse the processes described by differential_equations in the form of

        dX = drif(X, t) dt + diffusion(X, t) dW

    by reversing the drift and applying a correction:

        dX = [-drif(X, t) + correction] dt + diffusion(X, t) dW

    The correction is proportional to the square of diffusion

    Parameters
    ----------
    sde : stochastic_differential_equation
        The stochastic differential_equations to be reversed

    drift_corrector : direction_calculator
        A function that takes X as input and output the drift correction
    """


    def __init__(
        self,
        sde: stochastic_differential_equation,
        drift_corrector: direction_calculator
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
