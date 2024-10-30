"""
dataprocessing.interfaces
=========================

Provides the interface for all data transformation in this package

Class
-----
Transform       : A callable object that acts as a black-box transformation
"""


from abc import ABC, abstractmethod
from typing import Any


class Transform(ABC):
    """
    A callable object that acts as a black-box transformation

    Methods
    -------
    `__call__(self, data)`
        Transform `data` into `Transform(data)`
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
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
