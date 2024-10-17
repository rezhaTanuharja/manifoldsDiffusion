"""
eulerian
========

An approach to diffusion that treats data points as a continuum

Interfaces
----------
stochasticprocesses     : The interface of all stochastic processes in this package
"""

from . import stochasticprocesses


__all__ = [
    'stochasticprocesses',
]
