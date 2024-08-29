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

from ..timeintegrators import FirstOrder


class DataRecorder(ABC):
    """
    An interface to record data

    Methods
    -------
    reset(inverse_value_problems, num_samples)
        Reset and prepare to store a number of solution to inverse value problems

    store(problem_index, result)
        Define what to do with the result of an inverse value problem

    get_record()
        Provide access to previously recorded data
    """


    @abstractmethod
    def reset(self, inverse_value_problems, num_samples) -> None:
        """
        Reset instance to prepare for a new data recording

        Parameters
        ----------
        inverse_value_problems : Tuple()
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def store(self, problem_index, result) -> None:
        """
        Define what to do with a data chunk, e.g., store in record or do nothing

        Parameters
        ----------
        data_chunk : One piece of data chunk
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
        self.time_integrator = time_integrator
        self.data_recorder = data_recorder


    @abstractmethod
    def get_samples(self, *args, **kwargs):
        """
        Extract samples by solving SDEs

        Parameters
        ----------
        sde : StochasticDifferentialEquation
            The SDE to solve

        Returns
        -------
        torch.Tensor
            The sampled solution of sde
        """
        raise NotImplementedError("Subclasses must implement this method")
