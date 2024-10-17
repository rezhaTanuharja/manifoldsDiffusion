"""
eulerian.stochasticprocesses.univariate.functions
=================================================

Provides functionalities to define CDFs

Modules
-------
periodic    : Provide CDFs that are solutions to PDEs with periodic boundary conditions

Interfaces
----------
CumulativeDistributionFunction
    The interface for all distribution functions in this package
"""

from .interfaces import CumulativeDistributionFunction

from . import periodic


__all__ = [
    'CumulativeDistributionFunction',
    'periodic',
]
