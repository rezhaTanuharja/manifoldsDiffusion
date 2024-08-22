"""
diffusionmodels
===============

A package that provides functionalities to define diffusion models in manifolds.

To create a diffusion model in a manifold, one must define:
    - A manifold structure, particularly the exponential and log maps
    - A stochastic differential equation that governs the diffusion process
    - A time integration method to solve the SDEs numerically
    - A sampler to store SDEs solution at discrete points
    - A score function to reverse the diffusion process

Modules
-------
manifolds               : Provides various manifold structures
differentialequations   : Provides various stochastic differential equations
timeintegrators         : Provides various time integration methods
samplers                : Provides various sampling strategies
scorefunctions          : Provides various type of score functions
"""


from . import manifolds
from . import differentialequations
from . import timeintegrators
from . import samplers
from . import scorefunctions
from . import utilities


__all__ = [
    'manifolds',
    'differentialequations',
    'timeintegrators',
    'samplers',
    'scorefunctions',
    'utilities'
]
