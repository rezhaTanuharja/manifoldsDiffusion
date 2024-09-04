"""
diffusionmodels.solvers
=======================

Provides functionalities to solve SDE and store a finite number of the solution

Modules
-------
baseclass           : Provides the interfaces of all solvers in this package
simpleclass         : Implements various simple solvers
"""


from .baseclass import Solver
from .simpleclass import SimpleSolver


__all__ = [
    'Solver',
    'SimpleSolver'
]
