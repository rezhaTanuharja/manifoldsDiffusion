"""
dataprocessing
==============

Provides functionalities to define data processing steps

Interfaces
----------
Transform       : A callable object that acts as a black-box transformation

Classes
-------
Pipeline        : A transformation that consist of a sequence of transformations
"""


from .interfaces import Transform
from .simpleclass import Pipeline


__all__ = [
    'Transform',
    'Pipeline',
]
