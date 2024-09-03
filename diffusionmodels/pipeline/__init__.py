"""
pipeline
========

Provides functionalities to define a data transformation pipeline

Modules
-------
baseclass       : Provides the interface for all transformations in this package
simpleclass     : Implements various simple transformations
"""


from .baseclass import Transform
from .simpleclass import Pipeline


__all__ = [
    'Transform',
    'Pipeline',
]
