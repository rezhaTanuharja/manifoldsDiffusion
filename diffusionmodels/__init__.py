"""
diffusionmodels
===============

A package that provides functionalities to define diffusion models in manifolds.

In general, there are two main components of a diffusion model:

    - Two initial-value problems:
        A forward noising process
        A backward denoising process
        
    - A stochastic differential equation solver

To define an initial-value problem for a diffusion model one needs to define:

    - A manifold structure
    - An initial condition
    - A stochastic differential equation

To define a stochastic differential equation solver one needs to define:
    
    - A time integration method
    - A data recorder to store the solution

Modules
-------
initialvalueproblems    : Provides functionalities to define initial value problems
solvers                 : Provides functionalities to solve initial value problems
"""


from . import manifolds
from . import differentialequations
from . import timeintegrators
from . import samplers


__all__ = [
    'manifolds',
    'differentialequations',
    'timeintegrators',
    'samplers'
]
