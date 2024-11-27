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


from .interfaces import Manifold
from . import rotationalgroups


__all__ = [
    'Manifold',
    'rotationalgroups',
]
