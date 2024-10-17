"""
eulerian.stochasticprocesses.univariate.inversion
=================================================

Provides functionalities to find the inverse of CDFs to sample via the inverse transform method

interfaces
----------
InversionMethod : The interface for all inversion methods in the Eulerian module

Classes
-------
Bisection       : An iterative root-finder via bisection
"""

from .interfaces import InversionMethod
from .simpleclass import Bisection


__all__ = [
    'InversionMethod',
    'Bisection',
]
