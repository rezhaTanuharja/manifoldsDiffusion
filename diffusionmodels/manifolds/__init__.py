"""
manifolds.structures
====================

A module that provides functionalities to define manifold structures.
"""


from .baseclass import Manifold
from .simpleclass import Euclidean
from .matrixclass import SO3


__all__ = [
    'Manifold',
    'Euclidean',
    'SO3'
]
