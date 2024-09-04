import torch

from .baseclass import DataRecorder


class SimpleRecorder(DataRecorder):
    """
    A class of data recorder that simply store all results given to it

    Private Attributes
    ------------------
    `_device : torch.device`
        The device that stores the record

    `_records : List[torch.Tensor]`
        A list of tensors to store data

    `_indices : List[int]`
        A list of indices of the last saved data in each tensor inside _records
    """

    def __init__(self):
        self._device = torch.device('cpu')

    def to(self, device = torch.device) -> None:
        self._device = device

    def reset(
        self,
        initial_value: torch.Tensor,
        num_samples: int
    ) -> None:

        self._indices = 0

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
        self._records[self._indices] = result
        self._timestamps[self._indices] = time
        self._indices += 1

    def get_record(self):
        return {
            'data': self._records,
            'time': self._timestamps
        }
