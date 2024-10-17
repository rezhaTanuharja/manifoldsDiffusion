"""
eulerian.stochasticprocesses.univariate
=======================================

Implements various univariate stochastic processes

Modules
-------
functions           : Provides functionalities to define CDFs
inversion           : Provides functionalities to invert CDFs

Classes
-------
InverseTransform    : A stochastic process that is defined by its CDF
"""

from . import functions
from . import inversion

from .simpleclass import InverseTransform


__all__ = [

    'functions',
    'inversion',

    'InverseTransform',
]
