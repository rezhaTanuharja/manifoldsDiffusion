"""
distributions.inversion
=======================

Provides functionalities to find the inverse of CDFs to sample via the inverse transform method

Modules
-------
baseclass       : Provides the interface for all inversion method in this package
simpleclass     : Implements various simple inversion methods
"""

from .baseclass import InversionMethod
from .simpleclass import Bisection


__all__ = [
    'InversionMethod',
    'Bisection',
]
