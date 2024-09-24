"""
distributions
=============

Provides functionalities to define a data distribution

Modules
-------
baseclass       : Provides the interface for all distribution in this package
simpleclass     : Provides various simple distributions

Submodules
----------
functions       : Provides distribution functions that include PDFs and CDFs
inversion       : Provides methods to invert a CDF to sample with the inverse transform method
"""

from . import functions
from . import inversion

from .baseclass import Distribution
from .simpleclass import MultivariateGaussian, InverseTransform


__all__ = [
    'functions',
    'inversion',
    'Distribution',
    'MultivariateGaussian',
    'InverseTransform',
]
