import pytest
import torch
from ....manifolds import rotationalgroups


@pytest.fixture(scope="class")
def manifold_cpu_float():
    return rotationalgroups.SpecialOrthogonal3(data_type=torch.float32)


@pytest.mark.cpu
class TestCPUOperationsFloat:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

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

        try:
            new_points = manifold_cpu_float.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

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

        try:
            new_points = manifold_cpu_float.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

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
                    rtol=1e-6,
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

        try:
            results_1 = manifold_cpu_float.exp(
                manifold_cpu_float.exp(points, vectors_1), vectors_2
            )
            results_2 = manifold_cpu_float.exp(points, vectors_1 + vectors_2)

        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

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

        try:
            vectors = manifold_cpu_float.log(starts, ends)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

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

        try:
            vectors = manifold_cpu_float.log(starts, ends)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

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

        try:
            computed_vectors = manifold_cpu_float.log(
                points, manifold_cpu_float.exp(points, vectors)
            )
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

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
    return rotationalgroups.SpecialOrthogonal3(data_type=torch.float64)


@pytest.mark.cpu
class TestCPUOperationsDouble:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

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

        try:
            new_points = manifold_cpu_double.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

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

        try:
            new_points = manifold_cpu_double.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

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
                    rtol=1e-16,
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

        try:
            results_1 = manifold_cpu_double.exp(
                manifold_cpu_double.exp(points, vectors_1), vectors_2
            )
            results_2 = manifold_cpu_double.exp(points, vectors_1 + vectors_2)

        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

        assert results_1.shape == (2, 6, 3, 3)
        assert results_2.shape == (2, 6, 3, 3)

        for repetition in range(results_1.shape[0]):
            for distinct_point in range(results_1.shape[1]):
                assert torch.allclose(
                    results_1[repetition, distinct_point],
                    results_2[repetition, distinct_point],
                    rtol=1e-16,
                )

    def test_log_compatibility(self, manifold_cpu_double) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        starts = torch.eye(3, dtype=torch.float64).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, dtype=torch.float64).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        try:
            vectors = manifold_cpu_double.log(starts, ends)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

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

        try:
            vectors = manifold_cpu_double.log(starts, ends)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

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
                    rtol=5e-8,
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

        try:
            computed_vectors = manifold_cpu_double.log(
                points, manifold_cpu_double.exp(points, vectors)
            )
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    rtol=5e-8,
                )
