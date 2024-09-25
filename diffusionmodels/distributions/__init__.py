"""
diffusionmodels.distributions
=============================

Provides functionalities to define a data distribution

Modules
-------
univariate      : Implements various univariate distributions
multivariate    : Implements various multivariate distributions
"""

from .interfaces import Distribution

from . import univariate
from . import multivariate


__all__ = [
    
    'Distribution',

    'univariate',
    'multivariate'
]
