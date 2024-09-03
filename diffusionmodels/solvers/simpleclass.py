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
from typing import List, Tuple

from ..differentialequations import StochasticDifferentialEquation
from ..timeintegrators import Explicit
from ..recorders import DataRecorder

from .baseclass import Solver


class SimpleSolver(Solver):
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
        time_integrator: Explicit,
        data_recorder: DataRecorder,
        num_points: int,
        grid_size: float
    ) -> None:
        self._time_integrator = time_integrator
        self._data_recorder = data_recorder
        self._num_points = num_points
        self._grid_size = grid_size


    def set_num_points(self, num_points: int) -> None:
        self._num_points = num_points

    def set_grid_size(self, grid_size: float) -> None:
        self._grid_size = grid_size


    def solve(
        self,
        initial_value, stochastic_de
    ) -> torch.Tensor:

        self._data_recorder.reset(initial_value, self._num_points)

        X = initial_value

        for n in range(self._num_points):

            X = self._time_integrator.step_forward(

                stochastic_de = stochastic_de,
                X = X,
                t = n * self._grid_size,
                dt = self._grid_size

            )

            self._data_recorder.store(
                result = X,
                time = (n + 1) * self._grid_size)

        return self._data_recorder.get_record()
