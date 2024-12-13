"""
Implements univariate process defined by their CDFs

Classes
-------
`InverseTransform`
A stochastic process defined by a CDF and sampled using inverse transform method

Modules
-------
cdf             : implements various simple CDFs
rootfinders     : implements various simple `RootFinder`
"""

from typing import Tuple

import torch

from ...interfaces import DensityFunction, StochasticProcess
from ..interfaces import CumulativeDistributionFunction, RootFinder
from . import cdf, rootfinders


class InverseTransform(StochasticProcess):
    """
    A process defined by its CDF and sampled using the inverse transform method
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
        The CDF that defines the process, a callable object

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
            function=self._distribution,
            target_values=target_values,
            interval=(
                self._distribution.support["lower"],
                self._distribution.support["upper"],
            ),
        )


__all__ = ["cdf", "rootfinders", "InverseTransform"]
