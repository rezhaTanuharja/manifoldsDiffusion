"""
Provides functionalities to define data processing steps.

Interfaces
----------
`Transform`
The interface for all data processor in this project

Modules
-------
sequential  : define processes that consist of a sequence of transformations
"""


from .interfaces import Transform
from . import sequential


__all__ = [
    'Transform',
    'sequential',
]
