"""
diffusionmodels.processes
=========================

Provides functionalities to define forward and reverse diffusion processes

Modules
-------
eulerian        : deals with the evolution of data concentration
lagrangian      : deals with the trajectory of each data point
"""

from . import eulerian
from . import lagrangian


__all__ = [
    'eulerian',
    'lagrangian',
]
