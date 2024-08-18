"""
time_integrators
================

This module provides functionalities to integrate SDEs over time.

Modules
-------
base_class  : Defines the abstract class for time_integrators
"""


from .base_class import predictor


__all__ = [
    'predictor'
]
