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
from typing import List, Tuple

from ..differentialequations import StochasticDifferentialEquation
from ..timeintegrators import FirstOrder
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
        time_integrator: FirstOrder,
        data_recorder: DataRecorder,
        num_points: int,
        grid_size: float
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def solve(
        self,
        initial_value_problems: List[Tuple[torch.Tensor, StochasticDifferentialEquation]],
    ) -> List[torch.Tensor]:
        """
        Extract samples by solving SDEs

        Parameters
        ----------
        initial_value_problems : InitialValueProblems
            A list of dict of initial-value problems

        num_samples : int
            The number of solutions to record

        dt : float
            The time increment between consecutive solutions

        Returns
        -------
        List[torch.Tensor, ...]
            The sampled solutions for each initial-value problem
        """
        raise NotImplementedError("Subclasses must implement this method")
