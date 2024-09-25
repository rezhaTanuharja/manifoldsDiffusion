"""
diffusionmodels.utilities
=========================

Provides miscellaneous functionalities for diffusionmodels

Functions
---------
unused_variables    : A function to explicitly declare that variables are not used
"""


from .warningsuppressors import unused_variables


__all__ = [
    'unused_variables',
]
