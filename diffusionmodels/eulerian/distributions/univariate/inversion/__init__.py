"""
diffusionmodels.distributions.univariate.inversion
==================================================

Provides functionalities to find the inverse of CDFs to sample via the inverse transform method

Classes
-------
InversionMethod : The interface for all inversion methods in this package
Bisection       : Invert CDFs using a bisection root-finder
"""

from .interfaces import InversionMethod
from .simpleclass import Bisection, Newton, Secant


__all__ = [
    'InversionMethod',
    'Bisection',
    'Newton',
    'Secant',
]
