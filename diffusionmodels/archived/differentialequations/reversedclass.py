"""
differentialequations.reversedclass
===================================

This module implements classes of reversed differentialequations

Classes
-------
CorrectedNegative
    Represent reversal by multiplying the drift with -1 and applying correction
"""


#TODO: Currently the reversed class has only been patched for visualization
# however, it needs to be refactored for general cases

import torch

from typing import Any

from diffusionmodels.utilities.warningsuppressors import unused_variables

from .baseclass import StochasticDifferentialEquation
from ..manifolds import Manifold
# from ..scorefunctions import DirectionCalculator


class CorrectedNegative(StochasticDifferentialEquation):
    """
    This class reverses the processes described by differential_equations

        dX = drift(X, t) dt + diffusion(X, t) dW

    by inverting the drift direction and applying a correction:

        dX = [-drift(X, t) + correction(X, t)] dt + diffusion(X, t) dW

    The correction is proportional to the square of diffusion

    Private Attributes
    ------------------
    _stochastic_de : StochasticDifferentialEquation
        The stochastic differential_equations to be reversed

    _drift_corrector : DirectionCalculator
        A function that takes X as input and output the drift correction
    """


    def __init__(
        self,
        stochastic_de: StochasticDifferentialEquation,
        drift_corrector: Any
    ):
        self._stochastic_de = stochastic_de
        self._drift_corrector = drift_corrector

    def to(self, device: torch.device) -> None:
        unused_variables(device)
        pass

    def manifold(self) -> Manifold:
        return self._stochastic_de.manifold()


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

        Y = X.flatten(1)
        t_tensor = torch.tensor([t]).cuda()

        result = (1.0) * self._drift_corrector(Y, t_tensor)
        result = result.unflatten(-1, (-1, *(self.manifold().tangent_dimension())))
        # result = result.view(1, 52, 3)
        return result

        # return (
        #     -self._stochastic_de.drift(X, t) + (
        #         # self._stochastic_de.diffusion(X, t) ** 2
        #         1.0
        #     ) * self._drift_corrector(X, t)
        # )


    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        return self._stochastic_de.diffusion(X, t)
