"""
diffusionmodels.samplers.datarecorders
======================================

This module implements the DataRecorder class

Classes
-------
SimpleRecorder
    A data recorder that store all results given to it
"""


import torch
from typing import List

from .baseclass import DataRecorder


class SimpleRecorder(DataRecorder):
    """
    A class of data recorder that simply store all results given to it

    Methods
    -------
    reset(inverse_value_problems, num_samples)
        Prepare record to store a number of solutions for inverse value problems

    store(result)
        Store results in record at index then increment index

    get_record()
        Returns all data inside record

    Private Attributes
    ------------------
    _records : List[torch.Tensor]
        A list of tensors to store data

    _indices : List[int]
        A list of indices of the last saved data in each tensor inside _records
    """


    def reset(self, initial_value_problems, num_samples):
        """
        Set record to an empty tensor that can store N times of X, set index to 0

        Parameters
        ----------
        X : torch.Tensor
            A tensor, representing the structure of each data chunk to store in record
        
        N : int
            The number of data chunks to store in record
        """
        self._records = [
            torch.zeros(
                num_samples, *(problem['initial_condition'].shape),
                device = problem['initial_condition'].device
            ) for problem in initial_value_problems
        ]

        self._indices = [0 for _ in initial_value_problems]


    def store(self, problem_index, result):
        """
        Store result as one data chunk inside record

        Parameters
        ----------
        result : torch.Tensor
            A tensor that can be stored as one data chunk in record
        """
        self._records[problem_index][self._indices[problem_index]] = result
        self._indices[problem_index] += 1

    def get_record(self) -> List[torch.Tensor]:
        """
        A method to access data in record

        Returns
        -------
        torch.Tensor
            All data inside record
        """
        return self._records
