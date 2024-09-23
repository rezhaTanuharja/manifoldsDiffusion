"""
distributions
=============

Provides functionalities to define a data distribution

Modules
-------
functions       : Provides functions such as PDFs and CDFs
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
