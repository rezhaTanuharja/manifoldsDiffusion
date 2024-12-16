"""
Checks mathematic operations of the SpecialOrthogonal3 class.
"""

import pytest
import torch

from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3


@pytest.fixture(scope="class")
def manifold_cpu_float():
    return SpecialOrthogonal3(data_type=torch.float32)


class TestCPUOperationsFloat:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

    def test_get_dimension(self, manifold_cpu_float) -> None:
        """
        Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
        """

        dimension = manifold_cpu_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 2

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_get_tangent_dimension(self, manifold_cpu_float) -> None:
        """
        Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
        """

        tangent_dimension = manifold_cpu_float.tangent_dimension

        assert isinstance(tangent_dimension, tuple)
        assert len(tangent_dimension) == 1

        for entry in tangent_dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_exp_compatibility(self, manifold_cpu_float) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        points = torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3)
        points = points.repeat(5, 1, 1, 1, 1)

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

        new_points = manifold_cpu_float.exp(points, vectors)

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, rtol=1e-16)

    def test_exp_correctness(self, manifold_cpu_float) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

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

        new_points = manifold_cpu_float.exp(points, vectors)

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
                    rtol=5e-5,
                )

    def test_exp_nested(self, manifold_cpu_float) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

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

        results_1 = manifold_cpu_float.exp(
            manifold_cpu_float.exp(points, vectors_1), vectors_2
        )
        results_2 = manifold_cpu_float.exp(points, vectors_1 + vectors_2)

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):
            for distinct_point in range(results_1.shape[1]):
                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    rtol=5e-5,
                )

    def test_log_compatibility(self, manifold_cpu_float) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        starts = torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, dtype=torch.float32).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        vectors = manifold_cpu_float.log(starts, ends)

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_cpu_float) -> None:
        """
        Checks that `log` produces correct results for small angles
        """

        starts = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
        starts = starts.repeat(2, 1, 1, 1)

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

        vectors = manifold_cpu_float.log(starts, ends)

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
                    rtol=5e-5,
                )

    def test_exp_log_connection(self, manifold_cpu_float) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        points = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float32,
        )

        computed_vectors = manifold_cpu_float.log(
            points, manifold_cpu_float.exp(points, vectors)
        )

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    rtol=1e-6,
                )


@pytest.fixture(scope="class")
def manifold_cpu_double():
    return SpecialOrthogonal3(data_type=torch.float64)


class TestCPUOperationsDouble:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

    def test_get_dimension(self, manifold_cpu_double) -> None:
        """
        Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
        """

        dimension = manifold_cpu_double.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 2

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_get_tangent_dimension(self, manifold_cpu_double) -> None:
        """
        Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
        """

        tangent_dimension = manifold_cpu_double.tangent_dimension

        assert isinstance(tangent_dimension, tuple)
        assert len(tangent_dimension) == 1

        for entry in tangent_dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_exp_compatibility(self, manifold_cpu_double) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        points = torch.eye(3, dtype=torch.float64).view(1, 1, 1, 3, 3)
        points = points.repeat(5, 1, 1, 1, 1)

        vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float64,
        )

        print(vectors.shape)

        new_points = manifold_cpu_double.exp(points, vectors)

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, rtol=1e-16)

    def test_exp_correctness(self, manifold_cpu_double) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float64,
        )

        new_points = manifold_cpu_double.exp(points, vectors)

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
            dtype=torch.float64,
        )

        for repetition in range(new_points.shape[0]):
            for distinct_point in range(new_points.shape[1]):
                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    rtol=5e-5,
                )

    def test_exp_nested(self, manifold_cpu_double) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors_1 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            dtype=torch.float64,
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
            dtype=torch.float64,
        )

        results_1 = manifold_cpu_double.exp(
            manifold_cpu_double.exp(points, vectors_1), vectors_2
        )
        results_2 = manifold_cpu_double.exp(points, vectors_1 + vectors_2)

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):
            for distinct_point in range(results_1.shape[1]):
                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    rtol=5e-5,
                )

    def test_log_compatibility(self, manifold_cpu_double) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        starts = torch.eye(3, dtype=torch.float64).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, dtype=torch.float64).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        vectors = manifold_cpu_double.log(starts, ends)

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_cpu_double) -> None:
        """
        Checks that `log` produces correct results for small angles
        """

        starts = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)
        starts = starts.repeat(2, 1, 1, 1)

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
            dtype=torch.float64,
        )

        vectors = manifold_cpu_double.log(starts, ends)

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float64,
        )

        for repetition in range(vectors.shape[0]):
            for distinct_vector in range(vectors.shape[1]):
                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    rtol=5e-5,
                )

    def test_exp_log_connection(self, manifold_cpu_double) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        points = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float64,
        )

        computed_vectors = manifold_cpu_double.log(
            points, manifold_cpu_double.exp(points, vectors)
        )

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    rtol=1e-6,
                )


@pytest.fixture(scope="class")
def manifold_gpu_float():
    manifold = SpecialOrthogonal3(data_type=torch.float32)
    manifold.to(torch.device("cuda"))
    return manifold


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")
class TestGPUOperationsFloat:
    """
    A group of `SpecialOrthogonal3` tests to be performed on gpu
    """

    def test_get_dimension(self, manifold_gpu_float) -> None:
        """
        Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
        """

        dimension = manifold_gpu_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 2

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_get_tangent_dimension(self, manifold_gpu_float) -> None:
        """
        Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
        """

        tangent_dimension = manifold_gpu_float.tangent_dimension

        assert isinstance(tangent_dimension, tuple)
        assert len(tangent_dimension) == 1

        for entry in tangent_dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_exp_compatibility(self, manifold_gpu_float) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 1, 3, 3
        )
        points = points.repeat(5, 1, 1, 1, 1)

        vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        print(vectors.shape)

        new_points = manifold_gpu_float.exp(points, vectors)

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, rtol=1e-16)

    def test_exp_correctness(self, manifold_gpu_float) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        new_points = manifold_gpu_float.exp(points, vectors)

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
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        for repetition in range(new_points.shape[0]):
            for distinct_point in range(new_points.shape[1]):
                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    rtol=5e-5,
                )

    def test_exp_nested(self, manifold_gpu_float) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors_1 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
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
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        results_1 = manifold_gpu_float.exp(
            manifold_gpu_float.exp(points, vectors_1), vectors_2
        )
        results_2 = manifold_gpu_float.exp(points, vectors_1 + vectors_2)

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):
            for distinct_point in range(results_1.shape[1]):
                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    rtol=5e-5,
                )

    def test_log_compatibility(self, manifold_gpu_float) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        starts = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 1, 3, 3
        )
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 3, 3
        )
        ends = ends.repeat(4, 1, 1)

        vectors = manifold_gpu_float.log(starts, ends)

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_gpu_float) -> None:
        """
        Checks that `log` produces correct results for small angles
        """

        starts = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 3, 3
        )
        starts = starts.repeat(2, 1, 1, 1)

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
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        vectors = manifold_gpu_float.log(starts, ends)

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        for repetition in range(vectors.shape[0]):
            for distinct_vector in range(vectors.shape[1]):
                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    rtol=5e-5,
                )

    def test_exp_log_connection(self, manifold_gpu_float) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float32).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        computed_vectors = manifold_gpu_float.log(
            points, manifold_gpu_float.exp(points, vectors)
        )

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    rtol=1e-6,
                )


@pytest.fixture(scope="class")
def manifold_gpu_double():
    manifold = SpecialOrthogonal3(data_type=torch.float64)
    manifold.to(torch.device("cuda"))
    return manifold


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")
class TestGPUOperationsdouble:
    """
    A group of `SpecialOrthogonal3` tests to be performed on gpu
    """

    def test_get_dimension(self, manifold_gpu_double) -> None:
        """
        Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
        """

        dimension = manifold_gpu_double.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 2

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_get_tangent_dimension(self, manifold_gpu_double) -> None:
        """
        Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
        """

        tangent_dimension = manifold_gpu_double.tangent_dimension

        assert isinstance(tangent_dimension, tuple)
        assert len(tangent_dimension) == 1

        for entry in tangent_dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_exp_compatibility(self, manifold_gpu_double) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 1, 3, 3
        )
        points = points.repeat(5, 1, 1, 1, 1)

        vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        print(vectors.shape)

        new_points = manifold_gpu_double.exp(points, vectors)

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, rtol=1e-16)

    def test_exp_correctness(self, manifold_gpu_double) -> None:
        """
        Checks that `exp` produce the correct results
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        new_points = manifold_gpu_double.exp(points, vectors)

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
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        for repetition in range(new_points.shape[0]):
            for distinct_point in range(new_points.shape[1]):
                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    rtol=5e-5,
                )

    def test_exp_nested(self, manifold_gpu_double) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors_1 = torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, -1.0 * torch.pi, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, 4.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
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
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        results_1 = manifold_gpu_double.exp(
            manifold_gpu_double.exp(points, vectors_1), vectors_2
        )
        results_2 = manifold_gpu_double.exp(points, vectors_1 + vectors_2)

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):
            for distinct_point in range(results_1.shape[1]):
                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    rtol=5e-5,
                )

    def test_log_compatibility(self, manifold_gpu_double) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        starts = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 1, 3, 3
        )
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 3, 3
        )
        ends = ends.repeat(4, 1, 1)

        vectors = manifold_gpu_double.log(starts, ends)

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_gpu_double) -> None:
        """
        Checks that `log` produces correct results for small angles
        """

        starts = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 3, 3
        )
        starts = starts.repeat(2, 1, 1, 1)

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
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        vectors = manifold_gpu_double.log(starts, ends)

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        for repetition in range(vectors.shape[0]):
            for distinct_vector in range(vectors.shape[1]):
                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    rtol=5e-5,
                )

    def test_exp_log_connection(self, manifold_gpu_double) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        points = torch.eye(3, device=torch.device("cuda"), dtype=torch.float64).view(
            1, 1, 3, 3
        )
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            device=torch.device("cuda"),
            dtype=torch.float64,
        )

        computed_vectors = manifold_gpu_double.log(
            points, manifold_gpu_double.exp(points, vectors)
        )

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    rtol=1e-6,
                )
