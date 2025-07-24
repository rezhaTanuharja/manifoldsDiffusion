"""Provides functionalities to define a manifold.

interfaces
----------
`Manifold`
The interface for all manifolds in this project

`RiemannianManifold`
The interface for all Riemannian manifolds in this project

Modules
-------
rotationalgroups  : Groups of all orthogonal matrices with determinant 1
"""

from . import rotationalgroups
from .interfaces import Manifold, RiemannianManifold

__all__ = [
    "Manifold",
    "RiemannianManifold",
    "rotationalgroups",
]
