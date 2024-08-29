"""
initialvalueproblems
====================

Provides functionalities to define initial value problems for diffusion models

Modules
-------
manifolds               : Provides functionalities to define manifold structures
differentialequations   : Provides functionalities to define stochastic differential equations
scorefunctions          : Provides functionalities to define "score functions"
"""


import torch

from . import manifolds
from . import differentialequations
from . import scorefunctions


# -- available manifolds
SpecialOrthogonal3 = manifolds.SpecialOrthogonal3

# -- available differential equations
ExplodingVariance = differentialequations.ExplodingVariance

# -- available score functions
# TODO


class InitialValueProblems:
    """
    A class that behaves like a list of disctionaries of initial condition and SDE

    Methods
    -------
    append(initial_condition, stochastic_de)
        add the pair into the list
    """

    def __init__(self):
        self._elements = []

    def __getitem__(self, index: int):
        return self._elements[index]

    def __iter__(self):
        return iter(self._elements)

    def append(
        self,
        initial_condition: torch.Tensor,
        stochastic_de: differentialequations.StochasticDifferentialEquation
    ):
        if initial_condition.dim() < 3:
            raise ValueError("Initial condition must be a tensor with dimension higher than 3")

        if initial_condition.shape[-2:] != stochastic_de.manifold().dimension():
            raise ValueError("Initial condition and SDE don't lie in the same manifold")

        self._elements.append(
            {
                'initial_condition': initial_condition,
                'stochastic_de': stochastic_de
            }
        )


__all__ = [
    'InitialValueProblems',

    'manifolds',
    'differentialequations',
    'scorefunctions',
]
