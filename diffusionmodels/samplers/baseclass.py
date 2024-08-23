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

from ..timeintegrators import FirstOrder
from ..differentialequations import StochasticDifferentialEquation


class DataRecorder(ABC):
    """
    An interface to record data

    Methods
    -------
    reset(args, kwargs)
        Reset record to prepare for a new data recording

    store(data_chunk, args, kwargs)
        Define what to do with a data chunk, e.g., store in record or do nothing

    get_record(args, kwargs)
        Access previously recorded data
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Reset instance to prepare for a new data recording
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def store(self, data_chunk, *args, **kwargs) -> None:
        """
        Define what to do with a data chunk, e.g., store in record or do nothing

        Parameters
        ----------
        data_chunk : One piece of data chunk
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def get_record(self, *args, **kwargs) -> torch.Tensor:
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
    def get_samples(
        self,
        sde: StochasticDifferentialEquation,
        *args, **kwargs
    ):
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
