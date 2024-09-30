"""
eulerian.stochasticprocesses.univariate.inversion
=================================================

Provides functionalities to find the inverse of CDFs to sample via the inverse transform method

Classes
-------
InversionMethod : The interface for all inversion methods in the Eulerian module
Bisection       : An iterative root-finder via bisection
Newton          : An iterative root-finder that requires computing gradients
Secant          : A Newton-like method that does not requires computing gradients
"""

from .interfaces import InversionMethod
from .simpleclass import Bisection


__all__ = [
    'InversionMethod',
    'Bisection',
]
