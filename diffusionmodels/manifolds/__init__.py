"""
diffusionmodels.manifolds
=========================

Provides functionalities to define manifold structures.

Modules
-------
baseclass       : Defines abstract classes of manifolds
simpleclass     : Implements various common manifolds
matrixclass     : Implements various matrix groups
"""


from .baseclass import Manifold
from .simpleclass import Euclidean
from .matrixclass import SpecialOrthogonal3


__all__ = [
    'Manifold',
    'Euclidean',
    'SpecialOrthogonal3'
]
