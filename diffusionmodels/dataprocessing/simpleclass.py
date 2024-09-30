"""
diffusionmodels.dataprocessing.simpleclass
==========================================

Implements various simple data processing steps

Class
-----
Pipeline        : A transformation that consists of sequential smaller transformations
"""


from .interfaces import Transform
from typing import List, Any


class Pipeline(Transform):
    """
    A transformation that consists of several sequential transformations (a pipeline)
    """

    def __init__(self, transforms: List[Transform]) -> None:
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
