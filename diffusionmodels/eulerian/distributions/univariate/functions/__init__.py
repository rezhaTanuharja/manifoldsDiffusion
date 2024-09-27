"""
diffusionmodels.distributions.univariate.functions
==================================================

Provides functionalities to define functions like PDFs and CDFs

Modules
-------
periodic    : Provide solutions of PDEs with periodic boundary conditions

Classes
-------
CumulativeDistributionFunction
    The interface for all distribution functions in this package
"""

from .interfaces import CumulativeDistributionFunction

from . import periodic


__all__ = [
    'CumulativeDistributionFunction',
    'periodic',
]
