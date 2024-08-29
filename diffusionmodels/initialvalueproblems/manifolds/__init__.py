"""
manifolds
=========

Provides functionalities to define manifold structures.

Modules
-------
baseclass       : Provides the interface for all manifolds in this package
matrixclass     : Implements various matrix groups
"""


from .baseclass import Manifold
from .matrixclass import SpecialOrthogonal3


__all__ = [
    'Manifold',
    'SpecialOrthogonal3',
]
