from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from ..differentialequations import StochasticDifferentialEquation
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
        X,
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
    def store(self, result: torch.Tensor, time) -> None:
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
    def get_record(self):
        """
        A method to access previously recorded data
        """
        raise NotImplementedError("Subclasses must implement this method")


