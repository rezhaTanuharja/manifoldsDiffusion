"""
Provides unit test for the manifold module.

Author
------
Rezha Adrian Tanuharja

Date
----
2024-08-01
"""


import pytest

from ...manifolds import Manifold, rotationalgroups

import jax
import jax.numpy as jnp


def test_construction() -> None:
    """
    Checks that `SpecialOrthogonal3` can be constructed as an instance of `Manifold`
    """

    try:
        manifold = rotationalgroups.SpecialOrthogonal3()

    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    assert isinstance(manifold, Manifold)


def test_get_dimension() -> None:
    """
    Checks that `dimension` can be invoked and return the tuple `(3, 3)`
    """

    try:
        manifold = rotationalgroups.SpecialOrthogonal3()
        dimension = manifold.dimension()
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    assert isinstance(dimension, tuple)
    assert len(dimension) == 2

    for entry in dimension:
        assert isinstance(entry, int)
        assert entry == 3


def test_get_tangent_dimension() -> None:
    """
    Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
    """

    try:
        manifold = rotationalgroups.SpecialOrthogonal3()
        tangent_dimension = manifold.tangent_dimension()
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    assert isinstance(tangent_dimension, tuple)
    assert len(tangent_dimension) == 1

    for entry in tangent_dimension:
        assert isinstance(entry, int)
        assert entry == 3


@pytest.fixture(scope = "class")
def manifold_cpu():
    return rotationalgroups.SpecialOrthogonal3()


@pytest.mark.cpu
class TestCPUOperations:
    """
    A group of manifold tests to be performed on CPU
    """


    def test_exp_compatibility(self, manifold_cpu) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """
        
        points = jnp.eye(N = 3, dtype = jnp.float32)
        points = points.reshape(1, 1, 1, 3, 3)
        points = points.repeat(5, axis = 0)

        vectors = jnp.array(

            object = [[
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],],

            dtype = jnp.float32

        )

        try:
            new_points = manifold_cpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert jnp.allclose(points, new_points, atol = 1e-12)
