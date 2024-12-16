"""
Implements various `CumulativeDistributionFunction` that are governed by the heat equation.

Classes
-------
`PeriodicHeatKernel`
"""

# from typing import Callable
#
# import torch
#
# from ...interfaces import CumulativeDistributionFunction
#
#
# class PeriodicHeatKernel(CumulativeDistributionFunction):
#     def __init__(
#         self,
#         num_waves: int,
#         mean_squared_displacement: Callable[[torch.Tensor], torch.Tensor],
#         data_type=torch.dtype,
#     ) -> None:
#         self._num_waves = num_waves
#         self._mean_squared_displacement = mean_squared_displacement
#         self._device = torch.device("cpu")
#         self._data_type = data_type
#
#     def to(self, device: torch.device) -> None:
#         self._device = device
