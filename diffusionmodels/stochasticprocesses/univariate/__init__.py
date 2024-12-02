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
cdf                 : implements various simple `CumulativeDistributionFunction`
rootfinders         : implements various simple `RootFinder`
"""


from .interfaces import CumulativeDistributionFunction, RootFinder
from . import inversetransform
from . import cdf
from . import rootfinders


__all__ = [

    'CumulativeDistributionFunction',
    'RootFinder',

    'inversetransform',
    'cdf',
    'rootfinders',
]
