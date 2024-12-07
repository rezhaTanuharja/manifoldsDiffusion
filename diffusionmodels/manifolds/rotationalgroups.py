"""
Provides group structures of orthogonal matrices with determinant 1.

Classes
-------
`SpecialOrthogonal3`
The 3-dimensional rotation matrix group
"""

from typing import Tuple

from .interfaces import Manifold

import torch


class SpecialOrthogonal3(Manifold):
    """
    The 3-dimensional rotation matrix group
    """

    def __init__(self, data_type: torch.dtype = torch.float32) -> None:
        """
        Construct an SO3 manifold with a given data type

        Parameters
        ----------
        `data_type: torch.dtype`
        The level of precision, e.g., torch.float32, torch.float64
        """

        # canonical basis of tangent space at the identity element
        self._bases = torch.tensor(
            [
                [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
            ],
            dtype=data_type,
        )

    def to(self, device: torch.device) -> None:
        self._bases = self._bases.to(device)

    @property
    def dimension(self) -> Tuple[int, ...]:
        return (3, 3)

    @property
    def tangent_dimension(self) -> Tuple[int, ...]:
        return (3,)

    def exp(self, points: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            points,
            torch.linalg.matrix_exp(
                # the skew matrices associated with the vectors
                torch.einsum("...j, jkl -> ...kl", vectors, self._bases)
            ),
        )

    # WARN: log mapping should only be used for points near each other

    def log(self, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        relative_rotation = torch.einsum("...ji, ...jk -> ...ik", starts, ends)

        angle = torch.arccos(
            torch.clip(
                0.5 * (torch.einsum("...ii -> ...", relative_rotation) - 1.0),
                min=-1.0,
                max=1.0,
            )
        )

        rotation_axis = torch.einsum(
            "...ijk, ...jk -> ...i", self._bases, relative_rotation
        )

        rotation_axis = rotation_axis / (
            torch.linalg.norm(rotation_axis, axis=-1, keepdims=True) + 1e-8
        )

        return torch.einsum("..., ...i -> ...i", angle, rotation_axis)
