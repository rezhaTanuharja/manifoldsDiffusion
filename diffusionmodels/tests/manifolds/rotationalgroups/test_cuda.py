import pytest
import torch

from ....manifolds import rotationalgroups


@pytest.fixture(scope="class")
def manifold_gpu_float():
    manifold = rotationalgroups.SpecialOrthogonal3(data_type=torch.float32)
    manifold.to(torch.device("cuda"))

    return manifold


@pytest.mark.gpu
class TestGPUOperationsFloat:
    """
    A group of manifold tests to be performed on GPU
    """

    def test_exp_compatibility(self, manifold_gpu_float) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 1, 1, 3, 3)
        points = points.repeat(5, 1, 1, 1, 1)

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
            new_points = manifold_gpu_float.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, atol=1e-16)

    def test_exp_correctness(self, manifold_gpu_float) -> None:
        """
        Checks that `exp` produce the correct results
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 1, 3, 3)
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
            device=gpu,
        )

        try:
            new_points = manifold_gpu_float.exp(points, vectors)
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
                    rtol=1e-6,
                )

    def test_exp_nested(self, manifold_gpu_float) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 1, 3, 3)
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
            results_1 = manifold_gpu_float.exp(
                manifold_gpu_float.exp(points, vectors_1), vectors_2
            )

            results_2 = manifold_gpu_float.exp(points, vectors_1 + vectors_2)

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

    def test_log_compatibility(self, manifold_gpu_float) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        gpu = torch.device("cuda")

        starts = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        try:
            vectors = manifold_gpu_float.log(starts, ends)
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
            device=gpu,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_gpu_float) -> None:
        gpu = torch.device("cuda")

        starts = torch.eye(3, dtype=torch.float32, device=gpu)
        starts = starts.view(1, 1, 3, 3)
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
            vectors = manifold_gpu_float.log(starts, ends)
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
            device=gpu,
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

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float32, device=gpu).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

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
            new_points = manifold_gpu_float.exp(points, vectors)
            computed_vectors = manifold_gpu_float.log(points, new_points)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_vector in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_vector],
                    vectors[distinct_vector],
                    rtol=1e-6,
                )


@pytest.fixture(scope="class")
def manifold_gpu_double():
    manifold = rotationalgroups.SpecialOrthogonal3(data_type=torch.float64)
    manifold.to(torch.device("cuda"))

    return manifold


@pytest.mark.gpu
class TestGPUOperations:
    """
    A group of manifold tests to be performed on GPU
    """

    def test_exp_compatibility(self, manifold_gpu_double) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 1, 1, 3, 3)
        points = points.repeat(5, 1, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=gpu,
        )

        try:
            new_points = manifold_gpu_double.exp(points, vectors)
        except Exception as e:
            assert False, f"Got exception when computing exp: {e}"

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, atol=1e-16)

    def test_exp_correctness(self, manifold_gpu_double) -> None:
        """
        Checks that `exp` produce the correct results
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 1, 3, 3)
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
            device=gpu,
        )

        try:
            new_points = manifold_gpu_double.exp(points, vectors)
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
            dtype=torch.float64,
            device=gpu,
        )

        for repetition in range(new_points.shape[0]):
            for distinct_point in range(new_points.shape[1]):
                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    rtol=1e-16,
                )

    def test_exp_nested(self, manifold_gpu_double) -> None:
        """
        Checks that nesting `exp` is equivalent to summing the vectors
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 1, 3, 3)
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
            dtype=torch.float64,
            device=gpu,
        )

        try:
            results_1 = manifold_gpu_double.exp(
                manifold_gpu_double.exp(points, vectors_1), vectors_2
            )

            results_2 = manifold_gpu_double.exp(points, vectors_1 + vectors_2)

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

    def test_log_compatibility(self, manifold_gpu_double) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """

        gpu = torch.device("cuda")

        starts = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        try:
            vectors = manifold_gpu_double.log(starts, ends)
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
            device=gpu,
        )
        assert torch.allclose(vectors, reference_vectors, rtol=1e-16)

    def test_log_correctness(self, manifold_gpu_double) -> None:
        gpu = torch.device("cuda")

        starts = torch.eye(3, dtype=torch.float64, device=gpu)
        starts = starts.view(1, 1, 3, 3)
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
            dtype=torch.float64,
            device=gpu,
        )

        try:
            vectors = manifold_gpu_double.log(starts, ends)
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
            device=gpu,
        )

        for repetition in range(vectors.shape[0]):
            for distinct_vector in range(vectors.shape[1]):
                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    rtol=5e-8,
                )

    def test_exp_log_connection(self, manifold_gpu_double) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """

        gpu = torch.device("cuda")

        points = torch.eye(3, dtype=torch.float64, device=gpu).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=torch.float64,
            device=gpu,
        )

        try:
            new_points = manifold_gpu_double.exp(points, vectors)
            computed_vectors = manifold_gpu_double.log(points, new_points)
        except Exception as e:
            assert False, f"Got exception when computing log: {e}"

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_vector in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_vector],
                    vectors[distinct_vector],
                    rtol=5e-8,
                )
