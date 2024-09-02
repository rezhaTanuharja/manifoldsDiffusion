"""
manifolds.matrixclass
===================== Implements various matrix groups Classes -------
SpecialOrthogonal3
    The 3-dimension rotation matrix group
"""


import torch
from typing import Tuple

from .baseclass import Manifold


class SpecialOrthogonal3(Manifold):
    """
    The 3-dimension rotation matrix group.

    Private Attributes
    ------------------
    _bases : torch.Tensor
        A bases of the tangent space at the identity element

    _dimension : Tuple[int, ...]
        The tensor shape of each point in the manifold

    _tangent_dimension : Tuple[int, ...]
        The tensor shape of each vector in the manifold tangent space
    """

    def __init__(self, device: torch.device) -> None:

        # -- use the canonical base of the skew symmetric space
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
        ]).to(device)

        self._dimension = (3, 3)
        self._tangent_dimension = (3,)


    def dimension(self) -> Tuple[int, ...]:
        return self._dimension


    def tangent_dimension(self) -> Tuple[int, ...]:
        return self._tangent_dimension


    def exp(self, X: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:

        # -- Compute skew symmetric matrix from axis angle representation
        skew_matrix = torch.einsum('...j, jkl -> ...kl', dX, self._bases)

        # -- Parallel transport the curve and increment X
        return torch.matmul(X, torch.linalg.matrix_exp(skew_matrix))


    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        # -- Compute trans(X)Y
        relative_rotation =  torch.einsum('...ji, ...jk -> ...ik', X, Y)

        # -- Compute geodesic distance, compensate for numerical inaccuracies
        angle = torch.acos(
            torch.clip(
                0.5 * (torch.einsum('...ii -> ...', relative_rotation) - 1.0),
                min = -1.0, max = 1.0
            )
        )

        # -- Compute axis of rotation
        axis = torch.stack([
            relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
            relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
            relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1]
        ], dim = -1)

        axis = axis / (torch.norm(axis, dim = -1, keepdim = True) + 1e-6)

        # -- Return the axis-angle representation
        return torch.einsum('..., ...i -> ...i', angle, axis)
