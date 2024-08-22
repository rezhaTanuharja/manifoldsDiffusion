"""
diffusionmodels.manifolds.matrixclass
=====================================

A module that implement matrix groups

Classes
-------
SO3
    The 3D rotational matrix group
"""


import torch

from .baseclass import Manifold


class SO3(Manifold):
    """
    The 3D rotational group

    Attributes
    ----------
    E : torch.Tensor
        A basis of the tangent space at the identity element

    Methods
    -------
    exp(X, dX)
        Increment a point X with a tangent vector dX
        
    log(X, Y)
        Calculate a tangent vector dX such that exp(X, dX) = Y
    """

    def __init__(self):
        self.E = torch.Tensor([
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


    def exp(self, X: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """
        Increment a point X with a tangent vector dX

        Parameters
        ----------
        X : torch.Tensor
            A point in the manifold

        dX : torch.Tensor
            A tangent vector in a tangent space on the manifold

        Returns
        -------
        torch.Tensor
            A point in the manifold
        """

        # -- Compute skew symmetric matrix from axis angle representation
        A = torch.einsum('...ij, jkl -> ...ikl', dX, self.E)

        # -- Parallel transport the curve
        return torch.matmul(X, torch.linalg.matrix_exp(A))


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
            A tangent vector on the manifold in the form of axis angle representation
        """

        # -- Compute transpose(X)Y
        R =  torch.einsum('...ji, ...jk -> ...ik', X, Y)

        # -- Compute geodesic distance
        theta = torch.acos(0.5 * (
                torch.einsum('...ii -> ...', R) - 1.0
            )
        )

        # -- Compute axis of rotation
        v = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim = -1)

        v = v / (torch.norm(v, dim = -1, keepdim = True) + 1e-6)

        # -- Return axis angle representation
        return torch.einsum('..., ...i -> ...i', theta, v)
