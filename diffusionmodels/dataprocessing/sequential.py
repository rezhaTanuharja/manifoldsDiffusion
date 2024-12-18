"""
Implements various sequential data transformations.

Classes
-------
`Pipeline`
A callable object that applies a sequence of transformations
"""

from typing import Any, Callable, List

from .interfaces import Transform


class Pipeline(Transform):
    """
    A callable object that applies a sequence of transformations
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
