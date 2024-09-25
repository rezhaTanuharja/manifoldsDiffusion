"""
diffusionmodels.distributions.univariate.inversion
==================================================

Provides functionalities to find the inverse of CDFs to sample via the inverse transform method

Classes
-------
Bisection       : Invert CDFs using a bisection root-finder
"""

from .baseclass import InversionMethod
from .simpleclass import Bisection


__all__ = [
    'InversionMethod',
    'Bisection',
]
