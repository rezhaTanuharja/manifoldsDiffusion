"""
Provides one-dimensional stochastic processes.

Interfaces
----------
`CumulativeDistributionFunction`
A purely abstract class that serves as an interface of all CDFs

`RootFinder`
A purely abstract class that serves as an interface of all root-finder

Modules
-------
uniform             : implements the univariate uniform process
inversetransforms   : implements univariate process defined by their CDFs
"""

from . import inversetransforms, periodic, uniform

__all__ = [
    "inversetransforms",
    "uniform",
    "periodic",
]
