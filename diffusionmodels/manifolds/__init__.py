"""
manifolds.structures
====================

A module that provides functionalities to define manifold structures.
"""


from .baseclass import Manifold
from .simpleclass import Euclidean


__all__ = [
    'Manifold',
    'Euclidean'
]
