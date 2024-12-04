"""
Provides unit test for the rotationalgroups manifolds module.

Author
------
Rezha Adrian Tanuharja

Date
----
2024-11-26
"""

import pytest
import torch


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


@pytest.fixture(scope="class")
def manifold_cpu():
    return rotationalgroups.SpecialOrthogonal3()


@pytest.mark.cpu
class TestCPUOperations:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

    def test_exp_compatibility(self, manifold_cpu) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        # an array with shape (5, 1, 1, 3, 3)
        points = torch.eye(3, dtype=torch.float32)
        points = points.reshape(1, 1, 1, 3, 3)
        points = points.repeat(5, *(1 for _ in points.shape[1:]))

        print(points.shape)

        # an array with shape (1, 4, 3)
        vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )

        print(vectors.shape)

        try:
            new_points = manifold_cpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, atol=1e-12)

    def test_exp_correctness(self, manifold_cpu) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = torch.eye(3, dtype=torch.float32)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        try:
            new_points = manifold_cpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (2, 6, 3, 3)

        reference_points = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [
                        0.0,
                        1.0 / 2**0.5,
                        -1.0 / 2**0.5,
                    ],
                    [0.0, 1.0 / 2**0.5, 1.0 / 2**0.5],
                ],
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                [
                    [-1.0 / 2**0.5, 0.0, 1.0 / 2**0.5],
                    [0.0, 1.0, 0.0],
                    [-1.0 / 2**0.5, 0.0, -1.0 / 2**0.5],
                ],
                [
                    [0.5, -0.5 * 3**0.5, 0.0],
                    [0.5 * 3**0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [-0.5, 0.5 * 3**0.5, 0.0],
                    [-0.5 * 3**0.5, -0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )

        for repetition in range(new_points.shape[0]):

            for distinct_point in range(new_points.shape[1]):

                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    atol=1e-12,
                )

    def test_exp_nested(self, manifold_cpu) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = torch.eye(3, dtype=torch.float32)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors_1 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        vectors_2 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.05 * torch.pi, 0.0, 0.0],
                [0.0, 1.0 * torch.pi, 0.0],
                [0.0, 0.33 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 2.5],
                [0.0, 0.0, 1.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        try:

            results_1 = manifold_cpu.exp(manifold_cpu.exp(points, vectors_1), vectors_2)

            results_2 = manifold_cpu.exp(points, vectors_1 + vectors_2)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):

            for distinct_point in range(results_1.shape[1]):

                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    atol=1e-3,
                )

    def test_log_compatibility(self, manifold_cpu) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        # an array with shape (5, 2, 1, 3, 3)
        starts = torch.eye(3, dtype=torch.float32)
        starts = starts.reshape(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, *(1 for _ in starts.shape[2:]))

        # an array with shape (4, 3, 3)
        ends = torch.eye(3, dtype=torch.float32)
        ends = ends.reshape(1, 3, 3)
        ends = ends.repeat(4, *(1 for _ in ends.shape[1:]))

        try:
            vectors = manifold_cpu.log(starts, ends)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(vectors, reference_vectors, atol=1e-12)

    def test_log_correctness(self, manifold_cpu) -> None:
        """
        Checks that `log` produces correct results for small angles
        """

        starts = torch.eye(3, dtype=torch.float32)
        starts = starts.reshape(1, 1, 3, 3)
        starts = starts.repeat(2, *(1 for _ in starts.shape[1:]))

        ends = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0 / 2**0.5, -1.0 / 2**0.5],
                    [0.0, 1.0 / 2**0.5, 1.0 / 2**0.5],
                ],
                [
                    [-1.0 / 2**0.5, 0.0, 1.0 / 2**0.5],
                    [0.0, 1.0, 0.0],
                    [-1.0 / 2**0.5, 0.0, -1.0 / 2**0.5],
                ],
                [
                    [0.5, -0.5 * 3**0.5, 0.0],
                    [0.5 * 3**0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [-0.5, 0.5 * 3**0.5, 0.0],
                    [-0.5 * 3**0.5, -0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )

        try:
            vectors = manifold_cpu.log(starts, ends)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        for repetition in range(vectors.shape[0]):

            for distinct_vector in range(vectors.shape[1]):

                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    atol=1e-3,
                )

    def test_exp_log_connection(self, manifold_cpu) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        points = torch.eye(3, dtype=torch.float32)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        try:
            computed_vectors = manifold_cpu.log(
                points, manifold_cpu.exp(points, vectors)
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):

            for distinct_point in range(computed_vectors.shape[1]):

                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    atol=1e-12,
                )


@pytest.fixture(scope="class")
def manifold_gpu():

    manifold = rotationalgroups.SpecialOrthogonal3()
    manifold.to(torch.device("cuda"))

    return manifold


@pytest.mark.gpu
class TestGPUOperations:
    """
    A group of manifold tests to be performed on GPU
    """

    def test_exp_compatibility(self, manifold_gpu) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        gpu = torch.device("cuda")

        # an array with shape (5, 1, 1, 3, 3)
        points = torch.eye(3, dtype=torch.float32, device=gpu)
        points = points.reshape(1, 1, 1, 3, 3)
        points = points.repeat(5, *(1 for _ in points.shape[1:]))

        # an array with shape (1, 4, 3)
        vectors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        try:
            new_points = manifold_gpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, atol=1e-12)

    def test_exp_correctness(self, manifold_gpu) -> None:
        """
        Checks that `exp` produce the correct results
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        try:
            new_points = manifold_gpu.exp(points, vectors)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert new_points.shape == (2, 6, 3, 3)

        reference_points = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0 / 2**0.5, -1.0 / 2**0.5],
                    [0.0, 1.0 / 2**0.5, 1.0 / 2**0.5],
                ],
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                [
                    [-1.0 / 2**0.5, 0.0, 1.0 / 2**0.5],
                    [0.0, 1.0, 0.0],
                    [-1.0 / 2**0.5, 0.0, -1.0 / 2**0.5],
                ],
                [
                    [0.5, -0.5 * 3**0.5, 0.0],
                    [0.5 * 3**0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [-0.5, 0.5 * 3**0.5, 0.0],
                    [-0.5 * 3**0.5, -0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        for repetition in range(new_points.shape[0]):

            for distinct_point in range(new_points.shape[1]):

                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    atol=1e-12,
                )

    def test_exp_nested(self, manifold_gpu) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors_1 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        vectors_2 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.05 * torch.pi, 0.0, 0.0],
                [0.0, 1.0 * torch.pi, 0.0],
                [0.0, 0.33 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 2.5],
                [0.0, 0.0, 1.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        try:

            results_1 = manifold_gpu.exp(manifold_gpu.exp(points, vectors_1), vectors_2)

            results_2 = manifold_gpu.exp(points, vectors_1 + vectors_2)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):

            for distinct_point in range(results_1.shape[1]):

                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    atol=1e-3,
                )

    def test_log_compatibility(self, manifold_gpu) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        gpu = torch.device("cuda")

        # an array with shape (5, 2, 1, 3, 3)
        starts = torch.eye(3, dtype=torch.float32, device=gpu)
        starts = starts.reshape(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, *(1 for _ in starts.shape[2:]))

        # an array with shape (4, 3, 3)
        ends = torch.eye(3, dtype=torch.float32, device=gpu)
        ends = ends.reshape(1, 3, 3)
        ends = ends.repeat(4, *(1 for _ in ends.shape[1:]))

        try:
            vectors = manifold_gpu.log(starts, ends)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
            device=gpu,
        )
        assert torch.allclose(vectors, reference_vectors, atol=1e-12)

    def test_log_correctness(self, manifold_gpu) -> None:

        gpu = torch.device("cuda")

        starts = torch.eye(3, dtype=torch.float32, device=gpu)
        starts = starts.reshape(1, 1, 3, 3)
        starts = starts.repeat(2, *(1 for _ in starts.shape[1:]))

        ends = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0 / 2**0.5, -1.0 / 2**0.5],
                    [0.0, 1.0 / 2**0.5, 1.0 / 2**0.5],
                ],
                [
                    [-1.0 / 2**0.5, 0.0, 1.0 / 2**0.5],
                    [0.0, 1.0, 0.0],
                    [-1.0 / 2**0.5, 0.0, -1.0 / 2**0.5],
                ],
                [
                    [0.5, -0.5 * 3**0.5, 0.0],
                    [0.5 * 3**0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [-0.5, 0.5 * 3**0.5, 0.0],
                    [-0.5 * 3**0.5, -0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        try:
            vectors = manifold_gpu.log(starts, ends)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        for repetition in range(vectors.shape[0]):

            for distinct_vector in range(vectors.shape[1]):

                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    atol=1e-3,
                )

    def test_exp_log_connection(self, manifold_gpu) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu)
        points = points.reshape(1, 1, 3, 3)
        points = points.repeat(2, *(1 for _ in points.shape[1:]))

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
            device=gpu,
        )

        try:
            new_points = manifold_gpu.exp(points, vectors)
            computed_vectors = manifold_gpu.log(points, new_points)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):

            for distinct_vector in range(computed_vectors.shape[1]):

                assert torch.allclose(
                    computed_vectors[repetition, distinct_vector],
                    vectors[distinct_vector],
                    atol=1e-12,
                )
