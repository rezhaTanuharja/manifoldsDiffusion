"""
poseestimation.dataloaders
==========================

Provides functionalities to load datasets

Modules
-------
tensorflowadaptor   : provides function to load TensorFlow datasets as a NumPy array generator
"""


from . import tensorflowadaptor


__all__ = [
    'tensorflowadaptor',
]
