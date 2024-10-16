"""
eulerian.stochasticprocesses
============================

Provides functionalities to define stochastic processes

Modules
-------
univariate      : Implements various univariate processes
multivariate    : Implements various multivariate processes
"""

from .interfaces import StochasticProcess

from . import univariate
from . import multivariate


__all__ = [
    
    'StochasticProcess',

    'univariate',
    'multivariate'
]