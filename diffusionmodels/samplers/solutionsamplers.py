"""
diffusionmodels.samplers.solutionsamplers
=========================================

This module implements various solution samplers

Classes
-------
SimpleSampler
    A class to sample solutions of SDEs with a fixed time increment
"""


import torch

from ..timeintegrators import FirstOrder
from ..differentialequations import Incrementor, StochasticDifferentialEquation

from .baseclass import SolutionSampler, DataRecorder


class SimpleSampler(SolutionSampler):
    """
    A class to sample solutions of SDEs with a fixed time increment

    Attributes
    ----------
    time_integrator : FirstOrder
        The time integration method used to solve SDEs

    data_recorder : DataRecorder
        The data recorder to define what and how solution is stored
    """

    def __init__(
        self,
        time_integrator: FirstOrder,
        incrementor: Incrementor,
        data_recorder: DataRecorder
    ) -> None:
        self.time_integrator = time_integrator
        self.incrementor = incrementor
        self.data_recorder = data_recorder


    def get_samples(
        self,
        sde: StochasticDifferentialEquation,
        initial_condition: torch.Tensor,
        num_samples: int,
        dt: float
    ) -> torch.Tensor:
        """
        Extract samples by solving SDEs

        Parameters
        ----------
        sde : StochasticDifferentialEquation
            The SDE to solve

        initial_condition : torch.Tensor
            The initial condition for the sde

        num_samples : int
            The number of solutions to compute

        dt : float
            The time increment between samples

        Returns
        -------
        torch.Tensor
            The sampled solution of sde
        """

        X = initial_condition

        self.data_recorder.reset(X, num_samples)

        for i in range(num_samples):
            # X = self.incrementor(X, self.time_integrator.step_forward(sde, X, i * dt, dt))
            X = self.time_integrator.step_forward(sde, X, i * dt, dt)
            self.data_recorder.store(X)

        return self.data_recorder.get_record()
