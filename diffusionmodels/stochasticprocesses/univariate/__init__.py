"""
Provides one-dimensional stochastic processes.

Modules
-------
inversetransforms   : implements univariate process defined by their CDFs
uniform             : implements the univariate uniform process
"""

from . import inversetransforms, uniform

__all__ = [
    "inversetransforms",
    "uniform",
]
