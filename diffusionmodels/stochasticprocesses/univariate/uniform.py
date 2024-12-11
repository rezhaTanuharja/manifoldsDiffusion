"""
Implements the univariate uniform probability density function.

Classes
-------
`Uniform`
A uniform probability density function defined on an interval
"""

from typing import Dict, Tuple

import torch

from ...utilities.warningsuppressors import unused_variables
from ..interfaces import DensityFunction


class Uniform(DensityFunction):
    """
    A uniform probability density function defined on an interval
    """

    def __init__(
        self, support: Dict[str, float], data_type: torch.dtype = torch.float32
    ) -> None:
        """
        Construct a density object for a univariate uniform distribution

        Parameters
        ----------
        `support: Dict[str, float]`
        A dictionary with keys 'lower' and 'upper', defines the interval

        `data_type: torch.dtype = torch.float32`
        The data type of floating points
        """
        self._data_type = data_type
        self._lower = support["lower"]
        self._upper = support["upper"]

    def to(self, device: torch.device) -> None:
        unused_variables(device)

    def at(self, time: torch.Tensor):
        unused_variables(time)
        return self

    @property
    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        lower_bound = torch.full_like(points, self._lower)
        upper_bound = torch.full_like(points, self._upper)

        values = 1.0 / (upper_bound - lower_bound)

        result = torch.where(
            (points < lower_bound) | (points > upper_bound),
            input=torch.zeros_like(points),
            other=values,
        )

        return result

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input=points, dtype=self._data_type)
