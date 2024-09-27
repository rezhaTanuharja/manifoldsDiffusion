"""
diffusionmodels.distributions.univariate
========================================

Implements various univariate stochastic processes

Modules
-------
functions           : Provides various functions for PDFs and CDFs
inversion           : Provides funtionalities to invert CDFs

Classes
-------
InverseTransform    : Distribution that can be sampled by inverting their CDFs
"""

from . import functions
from . import inversion

from .functions import CumulativeDistributionFunction
from .simpleclass import InverseTransform


__all__ = [

    'CumulativeDistributionFunction',

    'functions',
    'inversion',

    'InverseTransform',
]
