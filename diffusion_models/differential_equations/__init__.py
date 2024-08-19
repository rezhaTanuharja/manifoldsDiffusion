"""
diffusion_models.differential_equations
=======================================

This module provides functionalities to define differential equations.

Modules
-------
base_class  : Defines the abstract class for differential_equations
"""


from .base_class import stochastic_differential_equation
from .simple_class import standard_OU


__all__ = [
    'stochastic_differential_equation',
    'standard_OU'
]
