"""
diffusionmodels.timeintegrators.simpleclass
===========================================

Defines the simple classes for time integrators

Classes
-------
EulerMaruyama
    An explicit first-order time integrator

Heun
    A predictor-corrector time integrator using a trapezoidal rule
"""


import torch

from ..differentialequations import StochasticDifferentialEquation
from .baseclass import Explicit


class EulerMaruyama(Explicit):
    """
    An explicit first-order time integrator
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


class Heun(Explicit):
    """
    An explicit time-integation with trapezoidal predictor-corrector method

    Private Attributes
    ------------------
    _predictor : Explicit
        An explicit time integrator
    """


    def __init__(self, predictor: Explicit) -> None:
        self._predictor = predictor


    def step_forward(
        self,
        stochastic_de: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float, dt: float
    ) -> torch.Tensor:
        """
        Do a prediction, then correct the drift by simple averaging:

            (1) `X_pred = predictor.step_forward(X, t, dt)`

            (2) `drift = 0.5 * []`

            (3) `X(t + dt) = X(t) + drift dt + diffusion(X, t) dW`
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
