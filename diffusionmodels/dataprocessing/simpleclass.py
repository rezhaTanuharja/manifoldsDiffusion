"""
dataprocessing.simpleclass
==========================

Implements various simple data processing steps

Class
-----
Pipeline        : A transformation that consists of sequential smaller transformations
"""


from .interfaces import Transform
from typing import List, Any, Callable


class Pipeline(Transform):
    """
    A transformation that consists of a sequence of transformations
    """

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        """
        Parameters
        ----------
        `transforms: List[Transform]`
            A list of transformations to perform sequentially
        """
        self._transforms = transforms

    def __call__(self, data: Any) -> Any:
        
        for transform in self._transforms:
            data = transform(data)

        return data
