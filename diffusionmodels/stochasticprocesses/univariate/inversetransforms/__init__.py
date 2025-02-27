"""
Implements univariate processes defined by their CDFs.

Interfaces
----------
`CumulativeDistributionFunction`
The interface for all CDF in this project

`RootFinder`
The interface for all root finders of CDF

Classes
-------
`InverseTransform`
A stochastic process defined by a cumulativedistributions and sampled using inverse transform method

Modules
-------
cumulativedistributions     : implements various simple `CumulativeDistributionFunction`
rootfinders                 : implements various simple `RootFinder`
"""

from typing import Tuple

import torch

from ...interfaces import DensityFunction, StochasticProcess
from . import cumulativedistributions, rootfinders
from .interfaces import CumulativeDistributionFunction, RootFinder


class InverseTransform(StochasticProcess):
    """
    A process defined by its CDF and sampled using the inverse transform method
    """

    def __init__(
        self,
        distribution: CumulativeDistributionFunction,
        root_finder: RootFinder,
        device: torch.device = torch.device("cpu"),
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct an instance of `InverseTransform`

        Parameters
        ----------
        `distribution: CumulativeDistributionFunction`
        The cumulativedistributions that defines the process, a callable object

        `root_finder: RootFinder`
        A numerical solver to perform inverse transform sampling

        `device: torch.device = torch.device("cpu")`
        The hardware where tensor attributes reside

        `data_type: torch.dtype = torch.float32`
        The data type of all tensor class attributes
        """
        self._distribution = distribution
        self._root_finder = root_finder
        self._data_type = data_type
        self._time = torch.tensor(
            [
                0.0,
            ],
            device=device,
            dtype=data_type,
        )
        self._device = device

    def to(self, device: torch.device) -> None:
        self._device = device
        self._distribution.to(self._device)
        self._time = self._time.to(self._device)

    def at(self, time: torch.Tensor):
        self._time = time.to(self._device)
        self._distribution = self._distribution.at(time)
        return self

    @property
    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def density(self) -> DensityFunction:
        return self._distribution.gradient

    def sample(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        target_values = torch.rand(
            size=(*self._time.shape, num_samples),
            dtype=self._data_type,
            device=self._device,
        )

        return self._root_finder.solve(
            function=self._distribution,
            target_values=target_values,
            interval=(
                self._distribution.support["lower"],
                self._distribution.support["upper"],
            ),
        )


__all__ = [
    "cumulativedistributions",
    "rootfinders",
    "InverseTransform",
    "CumulativeDistributionFunction",
    "RootFinder",
]
