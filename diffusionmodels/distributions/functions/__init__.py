"""
distributions.functions
=======================

Provides functionalities to define functions like PDFs and CDFs

Modules
-------
baseclass       : Provides the interface for all PDFs and CDFs in this package
simpleclass     : Implements various simple PDFs and CDFs
"""

from .baseclass import DistributionFunction
from .simpleclass import Uniform

from . import periodic


__all__ = [
    'DistributionFunction',
    'Uniform',
    'periodic',
]
