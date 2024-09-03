"""
diffusionmodels.samplers.baseclass
==================================

This module defines the abstract base classes for samplers

Classes
-------
DataRecorder
    An abstract class to define how and which data is stored

SolutionSampler
    An abstract class to define how to extract a time-series from SDEs
"""


import torch
from abc import ABC, abstractmethod
from typing import Dict

from ..differentialequations import StochasticDifferentialEquation
from ..timeintegrators import Explicit
from ..recorders import DataRecorder


class Solver(ABC):
    """
    An abstract class to define how to extract a time-series from SDEs

    Methods
    -------
    get_samples(initial_value_problems, num_samples, dt)
        Solve initial-value problems and store a number of solutions
    """


    @abstractmethod
    def __init__(
        self,
        time_integrator: Explicit,
        data_recorder: DataRecorder,
        num_points: int,
        grid_size: float
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def set_num_points(self, num_points: int) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def set_grid_size(self, grid_size: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def solve(
        self,
        initial_value: torch.Tensor,
        stochastic_de: StochasticDifferentialEquation
    ) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Extract samples by solving SDEs

        Parameters
        ----------
        `initial_value : torch.Tensor`
            The initial value of the problem

        `stochastic_de : StochasticDifferentialEquation`
            The stochastic differential equation to solve

        Returns
        -------
        `Dict['time': torch.Tensor, 'data': torch.Tensor]`
            A dictionary of SDE solution and time
        """
        raise NotImplementedError("Subclasses must implement this method")
