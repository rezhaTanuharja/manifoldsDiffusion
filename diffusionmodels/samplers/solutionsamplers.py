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

    Private Attributes
    ------------------
    _time_integrator : FirstOrder
        The time integration method used to solve SDEs

    _data_recorder : DataRecorder
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
        initial_value_problems,
        num_samples: int,
        dt: float
    ) -> List[torch.Tensor]:

        self._data_recorder.reset(initial_value_problems, num_samples)

        for m, problem in enumerate(initial_value_problems):

            X = problem['initial_condition']

            for n in range(num_samples):
                X = self._time_integrator.step_forward(
                    problem['stochastic_de'], X, n * dt, dt
                )
                self._data_recorder.store(m, X)

        return self._data_recorder.get_record()
