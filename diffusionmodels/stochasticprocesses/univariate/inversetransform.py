"""
Implement univariate stochastic processes that are defined by their CDFs.

Classes
-------
`InverseTransform`
A stochastic process defined by a CDF and sampled using inverse transform method
"""

from .. import DensityFunction, StochasticProcess
from . import CumulativeDistributionFunction, RootFinder

from typing import Tuple, Self

import torch


class InverseTransform(StochasticProcess):
    """
    A process defined by its CDF and sampled using the inverse transform method
    """

    def __init__(
        self, distribution: CumulativeDistributionFunction, root_finder: RootFinder
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
        self._time = torch.tensor(
            [
                0.0,
            ]
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self._distribution.to(self._device)
        self._time.to(self._device)

    def at(self, time: torch.Tensor) -> Self:
        self._time = time
        self._time.to(self._device)
        return self

    def dimension(self) -> Tuple[int, ...]:
        return (1,)

    def density(self) -> DensityFunction:
        return self._distribution.gradient()

    def sample(
        self,
        num_samples: int,
    ) -> torch.Tensor:

        target_values = torch.rand(size=(*self._time.shape, num_samples))

        return self._root_finder.solve(
            function=self._distribution,
            target_values=target_values,
            interval=(
                self._distribution.support["lower"],
                self._distribution.support["upper"],
            ),
        )
