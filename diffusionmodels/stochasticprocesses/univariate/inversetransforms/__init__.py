"""
Implements univariate process defined by their cumulativedistributionss.

Classes
-------
`InverseTransform`
A stochastic process defined by a cumulativedistributions and sampled using inverse transform method

Modules
-------
cumulativedistributions     : implements various simple cumulativedistributionss
rootfinders                 : implements various simple `RootFinder`
"""

from typing import Tuple

import torch

from ...interfaces import DensityFunction, StochasticProcess
from . import cumulativedistributions, rootfinders
from .interfaces import CumulativeDistributionFunction, RootFinder


class InverseTransform(StochasticProcess):
    """
    A process defined by its cumulativedistributions and sampled using the inverse transform method
    """

    def __init__(
        self,
        distribution: CumulativeDistributionFunction,
        root_finder: RootFinder,
        data_type: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct an instance of InverseTransform

        Parameters
        ----------
        `distribution: CumulativeDistributionFunction`
        The cumulativedistributions that defines the process, a callable object

        `root_finder: RootFinder`
        A numerical solver to perform inverse transform sampling
        """
        self._distribution = distribution
        self._root_finder = root_finder
        self._data_type = data_type
        self._time = torch.tensor(
            [
                0.0,
            ],
            dtype=data_type,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> None:
        self._device = device
        self._distribution.to(self._device)
        self._time.to(self._device)

    def at(self, time: torch.Tensor):
        self._time = time
        self._time.to(self._device)
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
            size=(*self._time.shape, num_samples), dtype=self._data_type
        )

        return self._root_finder.solve(
            function=lambda points: self._distribution(points, self._time),
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
