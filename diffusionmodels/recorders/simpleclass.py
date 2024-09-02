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


    def reset(self, X, num_samples):

        self._records = torch.zeros(
            (num_samples, *X.shape),
            device = X.device
        )

        self._timestamps = torch.zeros(
            num_samples, device = X.device
        )

        self._indices = 0


    def store(self, result, time):
        self._records[self._indices] = result
        self._timestamps[self._indices] = time
        self._indices += 1

    def get_record(self):
        return {
            'noised': self._records,
            'time': self._timestamps
        }
