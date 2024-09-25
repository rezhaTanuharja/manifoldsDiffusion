"""
diffusionmodels.manifolds
=========================

Provides functionalities to define manifold structures.

Classes
-------
Manifold            : The interface for all manifolds in this package
SpecialOrthogonal3  : The Lie Group of 3D rotational matrices
"""


from .baseclass import Manifold
from .simpleclass import SpecialOrthogonal3


__all__ = [
    'Manifold',
    'SpecialOrthogonal3',
]
