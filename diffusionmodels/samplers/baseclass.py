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
from typing import List

from ..differentialequations import InitialValueProblems
from ..timeintegrators import FirstOrder


class DataRecorder(ABC):
    """
    An interface to record data

    Methods
    -------
    reset(initial_value_problems, num_samples)
        Reset and prepare to store a number of solution to initial value problems

    store(problem_index, result)
        Define what to do with the result of an initial value problem

    get_record()
        Provide access to previously recorded data
    """


    @abstractmethod
    def reset(
        self,
        initial_value_problems: InitialValueProblems,
        num_samples: int
    ) -> None:
        """
        Reset instance to prepare for a new data recording

        Parameters
        ----------
        initial_value_problems : InitialValueProblems
            A list of dict of initial-value problems

        num_samples : int
            The number of samples to record
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def store(self, problem_index: int, result: torch.Tensor) -> None:
        """
        Define what to do with a data chunk, e.g., store in record or do nothing

        Parameters
        ----------
        problem_index : int
            The index of the current initial-value problem

        result : torch.Tensor
            The latest solution of the initial-value problem
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def get_record(self) -> List[torch.Tensor]:
        """
        A method to access previously recorded data
        """
        raise NotImplementedError("Subclasses must implement this method")


class SolutionSampler(ABC):
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
        data_recorder: DataRecorder
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def get_samples(
        self,
        initial_value_problems: InitialValueProblems,
        num_samples: int,
        dt: float
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
