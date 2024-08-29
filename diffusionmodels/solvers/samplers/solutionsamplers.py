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
from typing import List

from ..timeintegrators import FirstOrder

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
        data_recorder: DataRecorder
    ) -> None:
        self._time_integrator = time_integrator
        self._data_recorder = data_recorder


    def get_samples(
        self,
        IVPs,
        num_samples: int,
        dt: float
    ) -> List[torch.Tensor]:
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

        self._data_recorder.reset(IVPs, num_samples)

        for m, problem in enumerate(IVPs):

            X = problem['initial_condition']

            for n in range(num_samples):
                X = self._time_integrator.step_forward(
                    problem['stochastic_de'], X, n * dt, dt
                )
                self._data_recorder.store(m, X)

        # X = initial_condition
        #
        # self._data_recorder.reset(X, num_samples)
        #
        # for i in range(num_samples):
        #     # X = self.incrementor(X, self.time_integrator.step_forward(sde, X, i * dt, dt))
        #     X = self._time_integrator.step_forward(sde, X, i * dt, dt)
        #     self._data_recorder.store(X)

        return self._data_recorder.get_record()
