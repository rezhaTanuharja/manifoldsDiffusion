"""
manifolds
=========

Provides functionalities to define manifold structures.

interfaces
----------
Manifold            : The interface for all manifolds in this package

Classes
-------
SpecialOrthogonal3  : The Lie Group of 3D rotational matrices
"""


from .interfaces import Manifold
from .simpleclass import SpecialOrthogonal3


__all__ = [
    'Manifold',
    'SpecialOrthogonal3',
]
