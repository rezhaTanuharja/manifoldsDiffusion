"""
A package that provides functionalities to define diffusion models in Riemannian geometries.

Modules
-------
dataprocessing      : Provides functionalities to define data preprocessing steps
eulerian            : For diffusion models that deals with the evolution of data concentration
manifolds           : Provides manifold structures
utilities           : Provides various useful functions

Author
------
Rezha Adrian Tanuharja

Date
----
2024-08-01
"""

from . import dataprocessing
# from . import eulerian
from . import manifolds
from . import utilities


__all__ = [
    'dataprocessing',
    # 'eulerian',
    'manifolds',
    'utilities',
]
