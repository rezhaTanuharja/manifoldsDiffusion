import torch

from typing import List

from .baseclass import DataRecorder


class SimpleRecorder(DataRecorder):
    """
    A class of data recorder that simply store all results given to it

    Private Attributes
    ------------------
    _records : List[torch.Tensor]
        A list of tensors to store data

    _indices : List[int]
        A list of indices of the last saved data in each tensor inside _records
    """


    def reset(self, initial_value_problems, num_samples):

        self._records = [
            torch.zeros(
                num_samples, *(problem[0].shape),
                device = problem[0].device
            ) for problem in initial_value_problems
        ]

        self._indices = [0 for _ in initial_value_problems]


    def store(self, problem_index, result):
        self._records[problem_index][self._indices[problem_index]] = result
        self._indices[problem_index] += 1

    def get_record(self) -> List[torch.Tensor]:
        return self._records
