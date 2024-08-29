"""
differentialequations
=====================

This module provides functionalities to define stochastic differential equations.

Modules
-------
baseclass      : Provides the interface for SDEs in this package
simpleclass    : Implements various simple SDEs
reversedclass  : Implements various reversal of SDEs
"""


from .baseclass import StochasticDifferentialEquation
from .simpleclass import ExplodingVariance
from .reversedclass import CorrectedNegative


__all__ = [
    'StochasticDifferentialEquation',
    'ExplodingVariance',
    'CorrectedNegative',
]
