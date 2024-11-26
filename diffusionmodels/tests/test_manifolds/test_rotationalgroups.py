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
import jax.numpy as jnp

from ...manifolds import Manifold, rotationalgroups


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
        
        # an array with shape (5, 1, 1, 3, 3)
        points = jnp.eye(N = 3, dtype = jnp.float32)
        points = points.reshape(1, 1, 1, 3, 3)
        points = points.repeat(5, axis = 0)

        # an array with shape (1, 4, 3)
        vectors = jnp.array(

            object = [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],

            dtype = jnp.float32

        )

        try:
            new_points = manifold_cpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert jnp.allclose(points, new_points, atol = 1e-12)


    def test_exp_correctness(self, manifold_cpu) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = jnp.eye(N = 3, dtype = jnp.float32)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, axis = 0)

        vectors = jnp.array([
            [       jnp.pi,           0.0,                0.0],
            [0.25 * jnp.pi,           0.0,                0.0],
            [          0.0, -1.0 * jnp.pi,                0.0],
            [          0.0, 0.75 * jnp.pi,                0.0],
            [          0.0,           0.0,       jnp.pi / 3.0],
            [          0.0,           0.0, 4.0 * jnp.pi / 3.0],
        ])

        try:
            new_points = manifold_cpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (2, 6, 3, 3)

        reference_points = jnp.array([
            [
                [ 1.0, 0.0, 0.0],
                [ 0.0,-1.0, 0.0],
                [ 0.0, 0.0,-1.0],
            ], [
                [ 1.0, 0.0,                             0.0],
                [ 0.0, 1.0 / jnp.sqrt(2),-1.0 / jnp.sqrt(2)],
                [ 0.0, 1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
            ], [
                [-1.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0],
                [ 0.0, 0.0,-1.0],
            ], [
                [-1.0 / jnp.sqrt(2), 0.0,  1.0 / jnp.sqrt(2)],
                [               0.0, 1.0,                0.0],
                [-1.0 / jnp.sqrt(2), 0.0, -1.0 / jnp.sqrt(2)],
            ], [
                [               0.5, -0.5 * jnp.sqrt(3), 0.0],
                [ 0.5 * jnp.sqrt(3),                0.5, 0.0],
                [ 0.0, 0.0, 1.0],
            ], [
                [              -0.5,  0.5 * jnp.sqrt(3), 0.0],
                [-0.5 * jnp.sqrt(3),               -0.5, 0.0],
                [ 0.0, 0.0, 1.0],
            ],
        ])

        for repetition in range(new_points.shape[0]):

            for distinct_point in range(new_points.shape[1]):

                assert jnp.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    atol = 1e-12
                )


    def test_exp_nested(self, manifold_cpu) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = jnp.eye(N = 3, dtype = jnp.float32)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, axis = 0)

        vectors_1 = jnp.array([
            [       jnp.pi,           0.0,                0.0],
            [0.25 * jnp.pi,           0.0,                0.0],
            [          0.0, -1.0 * jnp.pi,                0.0],
            [          0.0, 0.75 * jnp.pi,                0.0],
            [          0.0,           0.0,       jnp.pi / 3.0],
            [          0.0,           0.0, 4.0 * jnp.pi / 3.0],
        ])

        vectors_2 = jnp.array([
            [       jnp.pi,           0.0,                0.0],
            [0.05 * jnp.pi,           0.0,                0.0],
            [          0.0,  1.0 * jnp.pi,                0.0],
            [          0.0, 0.33 * jnp.pi,                0.0],
            [          0.0,           0.0,       jnp.pi / 2.5],
            [          0.0,           0.0, 1.0 * jnp.pi / 3.0],
        ])

        try:

            results_1 = manifold_cpu.exp(
                manifold_cpu.exp(
                    points, vectors_1
                ), vectors_2
            )

            print(results_1)

            results_2 = manifold_cpu.exp(points, vectors_1 + vectors_2)
            print(results_2)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):

            for distinct_point in range(results_1.shape[1]):

                assert jnp.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    atol = 1e-3
                )


    def test_log_compatibility(self, manifold_cpu) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """
        
        # an array with shape (5, 2, 1, 3, 3)
        starts = jnp.eye(N = 3, dtype = jnp.float32)
        starts = starts.reshape(1, 1, 1, 3, 3)
        starts = starts.repeat(5, axis = 0)
        starts = starts.repeat(2, axis = 1)

        # an array with shape (4, 3, 3)
        ends = jnp.eye(N = 3, dtype = jnp.float32)
        ends = ends.reshape(1, 3, 3)
        ends = ends.repeat(4, axis = 0)

        try:
            vectors = manifold_cpu.log(starts, ends)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = jnp.array([[[0.0, 0.0, 0.0],],])
        assert jnp.allclose(vectors, reference_vectors, atol = 1e-12)


