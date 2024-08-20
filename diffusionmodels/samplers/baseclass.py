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
    This class defines how and which data is stored

    Attributes
    ----------
    record : torch.Tensor
        A tensor to store data

    index : int
        The index of that last saved data

    Methods
    -------
    reset(X, N)
        Set record to an empty tensor with size N x *(X.shape), set index to 0

    store(result)
        Define what to do with result, e.g., store in record or do nothing

    get_record()
        Returns all data in record
    """


    def __init__(self) -> None:
        self.record = None
        self.index = None


    def reset(
        self,
        X: torch.Tensor,
        N: int
    ) -> None:
        """
        Set record to an empty tensor that can store N times of X, set index to 0

        Parameters
        ----------
        X : torch.Tensor
            A tensor, representing the structure of each data chunk to store in record
        
        N : int
            The number of data chunks to store in record
        """
        self.record = torch.zeros(N, *(X.shape), device = X.device)
        self.index = 0


    @abstractmethod
    def store(self, result: torch.Tensor) -> None:
        """
        Defines what to do with result, which is a data chunk

        Parameters
        ----------
        result : torch.Tensor
            A tensor that can be stored as one data chunk in record
        """
        raise NotImplementedError("Subclasses must implement this method")


    def get_record(self) -> torch.Tensor:
        """
        A method to access data in record

        Returns
        -------
        torch.Tensor
            All data inside record
        """
        return self.record


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
