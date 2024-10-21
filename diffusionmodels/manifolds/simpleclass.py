"""
manifolds.simpleclass
=====================

Implements various simple Riemannian geometries

Classes
-------
SpecialOrthogonal3
    The 3-dimension rotation matrix group
"""


import torch
from typing import Tuple

from .interfaces import Manifold


class SpecialOrthogonal3(Manifold):
    """
    The 3-dimension rotation matrix group.
    """

    def __init__(self) -> None:

        # the canonical basis of the tangent space at the identity element
        self._bases = torch.Tensor([
            [
                [ 0,  0,  0],
                [ 0,  0, -1],
                [ 0,  1,  0]
            ],
            [
                [ 0,  0,  1],
                [ 0,  0,  0],
                [-1,  0,  0]
            ],
            [
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  0,  0]
            ]
        ])

        self._dimension = (3, 3)
        self._tangent_dimension = (3,)

    def to(self, device: torch.device) -> None:
        self._bases = self._bases.to(device)

    def dimension(self) -> Tuple[int, ...]:
        return self._dimension

    def tangent_dimension(self) -> Tuple[int, ...]:
        return self._tangent_dimension

    def exp(
        self,
        points: torch.Tensor,
        tangent_vectors: torch.Tensor
    ) -> torch.Tensor:

        # compute skew symmetric matrix from axis angle representation
        skew_matrices = torch.einsum('...j, jkl -> ...kl', tangent_vectors, self._bases)

        # parallel transport the curve and increment X
        return torch.matmul(points, torch.linalg.matrix_exp(skew_matrices))

    def log(self, origins: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:

        # compute inverse(destinations)origins
        relative_rotation =  torch.einsum('...ji, ...jk -> ...ik', origins, destinations)

        # compute geodesic distance, compensate for numerical inaccuracies
        angle = torch.acos(
            torch.clip(
                0.5 * (torch.einsum('...ii -> ...', relative_rotation) - 1.0),
                min = -1.0, max = 1.0
            )
        )

        # compute axis of rotation
        axis = torch.einsum('...ijk, ...jk -> ...i', self._bases, relative_rotation)
        axis = axis / (torch.norm(axis, dim = -1, keepdim = True) + 1e-6)

        # return the axis-angle representation
        return torch.einsum('..., ...i -> ...i', angle, axis)
