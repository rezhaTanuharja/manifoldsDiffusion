"""
diffusionmodels.distributions
=============================

Provides functionalities to define a data distribution

Modules
-------
univariate      : Implements various univariate distributions
multivariate    : Implements various multivariate distributions
"""

from .interfaces import StochasticProcess

from . import univariate
from . import multivariate


__all__ = [
    
    'StochasticProcess',

    'univariate',
    'multivariate'
]
