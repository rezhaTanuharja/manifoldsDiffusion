"""
differentialequations
=====================

Provides functionalities to define stochastic differential equations.

Modules
-------
baseclass      : Provides the interface of all SDEs
simpleclass    : Implements various simple SDEs
reversedclass  : Implements various reversal of SDEs
"""


from .baseclass import StochasticDifferentialEquation, InitialValueProblems
from .simpleclass import ExplodingVariance
from .reversedclass import CorrectedNegative


__all__ = [
    'StochasticDifferentialEquation',
    'InitialValueProblems',
    'ExplodingVariance',
    'CorrectedNegative',
]
