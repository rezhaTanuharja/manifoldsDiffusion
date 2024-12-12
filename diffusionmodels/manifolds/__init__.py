"""
Provides functionalities to define data structures.

interfaces
----------
`Manifold`
The interface for all manifolds in this project

Modules
-------
rotationalgroups  : Groups of all orthogonal matrices with determinant 1
"""

from . import rotationalgroups
from .interfaces import Manifold

__all__ = [
    "Manifold",
    "rotationalgroups",
]
