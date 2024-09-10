import torch
from .baseclass import DataRecorder
from typing import Dict


class SimpleRecorder(DataRecorder):
    """
    A class of data recorder that simply store all results given to it

    Private Attributes
    ------------------
    `_device : torch.device`
        The device that stores the record

    `_records : torch.Tensor`
        A tensor to store data

    `_timestamps : torch.Tensor`
        The accompanying timestamps of data in the record

    `_current_index : int`
        This tracks the number of times a data chunk has been recorded
    """

    def __init__(self) -> None:
        self._device = torch.device('cpu')

    def to(self, device = torch.device) -> None:
        self._device = device

    def reset(self, initial_value: torch.Tensor, num_samples: int) -> None:

        self._current_index = 0

        self._timestamps = torch.zeros(
            size = (num_samples,), device = self._device
        )

        record_shape = (num_samples, ) + initial_value.shape

        self._records = torch.zeros(
            size = record_shape,
            dtype = initial_value.dtype,
            device = self._device
        )

    def store(self, result, time):
        self._records[self._current_index] = result
        self._timestamps[self._current_index] = time
        self._current_index += 1

    def get_record(self) -> Dict[str, torch.Tensor]:
        return {
            'data': self._records,
            'time': self._timestamps
        }


class UniformRandomRecorder(DataRecorder):
    """
    A recorder that stores only a uniform random subset of the data

    Private Attributes
    ------------------
    `_device : torch.device`
        The device that stores the record

    `_keep_ratio : float`
        The approximate ratio of subset in record to the full data

    `_records : torch.Tensor`
        A tensor to store data

    `_timestamps : torch.Tensor`
        The accompanying timestamps of data in record

    `_store_index : int`
        This tracks the number of times the store method has been called

    `_current_index : int`
        This tracks the number of times a data chunk has been recorded
    """

    def __init__(self, keep_ratio: float) -> None:
        self._device = torch.device('cpu')
        self._keep_ratio = keep_ratio

    def to(self, device = torch.device) -> None:
        self._device = device

    def reset(self, initial_value: torch.Tensor, num_samples: int) -> None:

        num_samples_kept = int(num_samples * self._keep_ratio)
        record_shape = (num_samples_kept, ) + initial_value.shape

        self._timestamps = torch.zeros(
            size = (num_samples_kept,), device = self._device
        )

        self._records = torch.zeros(
            size = record_shape,
            dtype = initial_value.dtype,
            device = self._device
        )

        # self._kept_indices = torch.randint(
        #     low = 0, high = num_samples,
        #     size = (num_samples_kept, )
        # )
        # self._kept_indices = torch.sort(self._kept_indices)
        self._kept_indices = torch.randperm(num_samples)[:num_samples_kept]


        self._store_index = 0
        self._current_index = 0

    def store(self, result: torch.Tensor, time: float) -> None:

        if self._store_index in self._kept_indices:

            self._records[self._current_index] = result
            self._timestamps[self._current_index] = time
            self._current_index += 1

        self._store_index += 1

    def get_record(self) -> Dict[str, torch.Tensor]:
        return {
            'data': self._records,
            'time': self._timestamps
        }
