"""
lagrangian.differentialequations
================================

Provides functionalities to define stochastic differential equations.

Classes
-------
StochasticDifferentialEquation
    The interface for all SDEs in this package

ExplodingVariance
    A forward SDE with zero drift

CorrectedNegative
    A backward SDE that reverse the forward drift and apply correction
"""


from .interfaces import StochasticDifferentialEquation
from .forward import ExplodingVariance
from .reverse import CorrectedNegative


__all__ = [
    'StochasticDifferentialEquation',
    'ExplodingVariance',
    'CorrectedNegative',
]
