"""
Provides functionalities to define stochastic processes.

Interfaces
----------
`DensityFunction`
The interface for all density functions in this project

`StochasticProcess`
The interface for all stochastic processes in this project

Modules
-------
univariate      : provides one-dimensional stochastic processeses
multivariate    : provides multi-dimensional stochastic processeses
"""

from . import multivariate, univariate
from .interfaces import DensityFunction, StochasticProcess

__all__ = [
    "DensityFunction",
    "StochasticProcess",
    "univariate",
    "multivariate",
]
