"""
lagrangian.solvers.recorders.interfaces
=======================================

Classes
-------
DataRecorder    : The interface for all data recorders in this package
"""


from abc import ABC, abstractmethod
from typing import Dict

import torch

class DataRecorder(ABC):
    """
    An interface to record data

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    reset(initial_value, num_samples)
        Reset and prepare to store a data with size equals to `num_samples` of `initial_value`

    store(problem_index, result)
        Define what to do with the result of an initial value problem

    get_record()
        Provide access to previously recorded data

    Private Attributes
    ------------------
    `_device : torch.device`
        The device to store the record
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def to(self, device = torch.device) -> None:
        """
        Move any tensor attributes to device

        Parameters
        ----------
        `device: torch.device`
            A device object from PyTorch
        """
        raise NotImplementedError("Subclasses must implement this method")
    

    @abstractmethod
    def reset(
        self, initial_value: torch.Tensor, num_samples: int) -> None:
        """
        Reset instance to prepare for a new data recording

        Parameters
        ----------
        `initial_value : torch.Tensor`
            A tensor that has the same shape as one data chunk to store

        `num_samples : int`
            The number of data chunks to record
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def store(
        self, result: torch.Tensor, time: float) -> None:
        """
        Define what to do with a data chunk, e.g., store in record or do nothing

        Parameters
        ----------
        `result : torch.Tensor`
            A data chunk(s) to store

        `time : float`
            The timestamp associated with the data chunk(s)
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def get_record(self) -> Dict[str, torch.Tensor]:
        """
        A method to access previously recorded data

        Returns
        -------
        `Dict['time': torch.Tensor, 'data': torch.Tensor]`
            A dictionary of stored data and the timestamps
        """
        raise NotImplementedError("Subclasses must implement this method")


