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
inversetransform    : implements univariate process defined by their CDFs
"""


from .interfaces import CumulativeDistributionFunction, RootFinder
from . import inversetransform


__all__ = [

    'CumulativeDistributionFunction',
    'RootFinder',

    'inversetransform'
]
