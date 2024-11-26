"""
manifolds.rotationalgroups
==========================

Provides group structures of orthogonal matrices with determinant 1

Classes
-------
SpecialOrthogonal3
    The 3-dimensional rotation matrix group
"""


from typing import Tuple

from .interfaces import Manifold

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class SpecialOrthogonal3(Manifold):
    """
    The 3-dimensional rotation matrix group
    """
    

    def __init__(self) -> None:
        
        # canonical basis of tangent space at the identity element
        self._bases = jnp.array([
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
        ], dtype = jnp.float32)

        self._dimension = (3, 3)
        self._tangent_dimension = (3,)


    def to(self, device: jax.Device) -> None:
        self._bases = jax.device_put(self._bases, device = device)

    
    def dimension(self) -> Tuple[int, ...]:
        return self._dimension


    def tangent_dimension(self) -> Tuple[int, ...]:
        return self._tangent_dimension


    def exp(self, points: jnp.ndarray, vectors: jnp.ndarray) -> jnp.ndarray:

        return jnp.matmul(

            points,

            jsp.linalg.expm(
                # the skew matrices associated with the vectors
                jnp.einsum('...j, jkl -> ...kl', vectors, self._bases)
            )

        )

    
    def log(self, starts: jnp.ndarray, ends: jnp.ndarray) -> jnp.ndarray:

        relative_rotation = jnp.matmul(starts.T, ends)

        angle = jnp.arccos(

            jnp.clip(

                0.5 * jnp.trace(relative_rotation) - 1.0,

                # clip for numerical stability
                min = -1.0, max = 1.0

            )

        )

        rotation_axis = jnp.einsum(
            '...ijk, ...jk -> ...i', self._bases, relative_rotation
        )

        rotation_axis = rotation_axis / (
            jnp.linalg.norm(rotation_axis, axis = -1, keepdims = True) + 1e-8
        )

        return jnp.einsum('..., ...i -> ...i', angle, rotation_axis)
