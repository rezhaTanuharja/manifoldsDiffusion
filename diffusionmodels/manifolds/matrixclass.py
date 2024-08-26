"""
diffusionmodels.manifolds.matrixclass
=====================================

A module that implement matrix groups

Classes
-------
SpecialOrthogonal3
    The 3-dimension rotation matrix group
"""


import torch

from .baseclass import Manifold


class SpecialOrthogonal3(Manifold):
    """
    The 3-dimension rotation matrix group.

    Attributes
    ----------
    bases : torch.Tensor
        A bases of the tangent space at the identity element

    Methods
    -------
    exp(X, dX)
        Increment a point X with a tangent vector dX
        
    log(X, Y)
        Calculate a tangent vector dX such that exp(X, dX) = Y
    """

    def __init__(self) -> None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bases = torch.Tensor([
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


    def exp(self, X: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """
        Increment a point X with a tangent vector dX

        Parameters
        ----------
        X : torch.Tensor
            A point in the manifold

        dX : torch.Tensor
            A tangent vector in a tangent space AT THE IDENTITY ELEMENT.
            The tangent vector is in the axis-angle representation.

        Returns
        -------
        torch.Tensor
            A point in the manifold
        """

        # -- Compute skew symmetric matrix from axis angle representation
        skew_matrix = torch.einsum('...j, jkl -> ...kl', dX, self.bases)

        # -- Parallel transport the curve and increment X
        return torch.matmul(X, torch.linalg.matrix_exp(skew_matrix))


    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Computes a tangent vector dX such that exp(X, dX) = Y

        Parameters
        ----------
        X : torch.Tensor
            The origin point in the manifold, a rotational matrix

        Y : torch.Tensor
            The destination point in the manifold, a rotational matrix

        Returns
        -------
        torch.Tensor
            A tangent vector in the tangent space at the identity element.
            The tangent vector is in the axis-angle representation.
        """

        # -- Compute trans(X)Y
        relative_rotation =  torch.einsum('...ji, ...jk -> ...ik', X, Y)

        # -- Compute geodesic distance
        angle = torch.acos(
            torch.clip(
                0.5 * (torch.einsum('...ii -> ...', relative_rotation) - 1.0), min = -1.0, max = 1.0
            )
        )

        # -- Compute axis of rotation
        axis = torch.stack([
            relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
            relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
            relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1]
        ], dim = -1)

        axis = axis / (torch.norm(axis, dim = -1, keepdim = True) + 1e-6)
        if axis.isnan().any():
            print('yay')

        # -- Return the axis-angle representation
        return torch.einsum('..., ...i -> ...i', angle, axis)
