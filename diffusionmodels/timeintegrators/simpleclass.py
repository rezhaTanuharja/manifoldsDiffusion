"""
diffusionmodels.timeintegrators.simpleclass
===========================================

Defines the simple classes for time integrators

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
    """


    def step_forward(
        self,
        stochastic_de: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:

        return stochastic_de.manifold().exp(
            X, 
            dt * stochastic_de.drift(X, t) + (dt ** 0.5) * stochastic_de.diffusion(X, t)
        )


class Heun(FirstOrder):
    """
    A predictor-corrector time integrator using the trapezoidal rule.

    Private Attributes
    ------------------
    _predictor : FirstOrder
        A first order time integrator
    """


    def __init__(self, predictor: FirstOrder) -> None:
        self._predictor = predictor


    def step_forward(
        self,
        stochastic_de: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float, dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) using the trapezoidal rule
            
            dX = 0.5 (sde.drift(X, t) + sde.drift(X_pred, t + dt)) dt + diffusion(X, t) dW

        The value of X_pred is obtained from predictor.step_forward(sde, X, t, dt)
        """

        # -- Predict future X using the Euler_Maruyama method
        X_pred = self._predictor.step_forward(stochastic_de, X, t, dt)

        # -- Use trapezoidal rule to correct the prediction
        return stochastic_de.manifold().exp(
            X, (
                0.5 * dt * (
                    stochastic_de.drift(X, t) + stochastic_de.drift(X_pred, t + dt)
                ) + (dt ** 0.5) * stochastic_de.diffusion(X, t)
            )
        )
