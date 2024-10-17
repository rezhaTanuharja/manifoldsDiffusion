"""
diffusionmodels.dataprocessing.interfaces
=========================================

Provides the interface for all data processing steps in this package

Class
-----
Transform       : An abstract class that serves as an interface for all transformation objects
"""


from abc import ABC, abstractmethod
from typing import Any


class Transform(ABC):
    """
    A callable object class to define pipeline for data processing

    Methods
    -------
    `__call__(self, data)`
        Transform the data and return the transformed data
    """


    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """
        The method that defines the transformation applied to data

        Parameters
        ----------
        `data : Any`

        Returns
        -------
        `Any`
            The transformed data, does not necessarily have the same type with the input
        """
        raise NotImplementedError("Subclasses must implement this method")
