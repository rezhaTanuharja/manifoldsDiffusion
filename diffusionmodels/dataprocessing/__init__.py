"""
dataprocessing
==============

Provides functionalities to define data processing steps

Modules
-------
baseclass       : Provides the interface for all data processing steps in this package
simpleclass     : Implements various simple data processing steps
"""


from .baseclass import Transform
from .simpleclass import Pipeline


__all__ = [
    'Transform',
    'Pipeline',
]
