"""
diffusionmodels.timeintegrators.simpleclass
===========================================

This module defines the simple classes for timeintegrators

Classes
-------
EulerMaruyama
    An implementation of an explicit time integrator

Heun
    An implementation of a predictor-corrector time integrator with a trapezoidal rule
"""


import torch

from ..differentialequations import StochasticDifferentialEquation
from .baseclass import FirstOrder


class EulerMaruyama(FirstOrder):
    """
    An explicit time integrator in the form of
        X(t + dt) = X(t) + dX(X(t), t)

    Methods
    -------
    step_forward(differential_equation, X, t, dt)
        Returns X(t) + dX(X(t), t)
    """

    def __init__(self) -> None:
        super().__init__()

    def step_forward(
        self,
        sde: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) from a given stochastic differential equation

        Parameters
        ----------
        sde: StochasticDifferentialEquation
            A differential equation in the form of
            dX = drift(X, t) dt + diffusion(X, t) dW

        X : torch.Tensor
            The current value, i.e., X(t)

        t : float
            The current time

        dt: float
            The time increment or the temporal step

        Returns
        -------
        torch.Tensor
            the predicted value of X(t + dt)
        """

        return X + dt * sde.drift(X, t) + torch.sqrt(torch.tensor(dt)) * sde.diffusion(X, t)


class Heun(FirstOrder):
    """
    A predictor-corrector time integrator using the trapezoidal rule.

    Parameters
    ----------
    predictor : first_order
        A first order time integrator

    Methods
    -------
    step_forward(differential_equation, X, t, dt)
        Returns X(t + dt)
    """

    def __init__(self, predictor: FirstOrder) -> None:
        super().__init__()
        self.predictor = predictor

    def step_forward(
        self,
        sde: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) using the trapezoidal rule
            
            dX = 0.5 (sde.drift(X, t) + sde.drift(X_pred, t + dt)) dt + diffusion(X, t) dW

        The value of X_pred is obtained from predictor.step_forward(sde, X, t, dt)

        Parameters
        ----------
        sde: StochasticDifferentialEquation
            A differential equation in the form of
            dX = drift(X, t) dt + diffusion(X, t) dW

        X : torch.Tensor
            The current value, i.e., X(t)

        t : float
            The current time

        dt: float
            The time increment or the temporal step

        Returns
        -------
        torch.Tensor
            the predicted value of X(t + dt)
        """

        # -- Predict future X using the Euler_Maruyama method
        X_pred = self.predictor.step_forward(sde, X, t, dt)

        # -- Use trapezoidal rule to correct the prediction
        return (
            X + 0.5 * dt * (
                sde.drift(X, t) + sde.drift(X_pred, t + dt)
            ) + torch.sqrt(torch.tensor(dt)) * sde.diffusion(X, t)
        )
