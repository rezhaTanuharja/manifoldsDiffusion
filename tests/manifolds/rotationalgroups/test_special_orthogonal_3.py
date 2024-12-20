"""
Checks mathematic operations of the SpecialOrthogonal3 class.
"""

from itertools import product

import pytest
import torch

from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3

devices = (
    torch.device("cpu"),
    torch.device("cuda", 0),
)

data_types_tolerances = zip((torch.float32, torch.float64), (1e-6, 1e-12))

test_parameters = [
    {
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "device": device,
    }
    for data_type_tolerance, device in product(data_types_tolerances, devices)
]


@pytest.fixture(params=test_parameters, scope="class")
def manifold_fixture(request):
    parameters = request.param

    manifold = SpecialOrthogonal3(data_type=parameters["data_type"])
    manifold.to(parameters["device"])

    return parameters, manifold


class TestSpecialOrthogonal3:
    """
    A group of `SpecialOrthogonal3` tests to be performed on CPU
    """

    def test_get_dimension(self, manifold_fixture) -> None:
        """
        Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
        """
        _, manifold = manifold_fixture

        dimension = manifold.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 2

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_get_tangent_dimension(self, manifold_fixture) -> None:
        """
        Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
        """
        _, manifold = manifold_fixture

        tangent_dimension = manifold.tangent_dimension

        assert isinstance(tangent_dimension, tuple)
        assert len(tangent_dimension) == 1

        for entry in tangent_dimension:
            assert isinstance(entry, int)
            assert entry == 3

    def test_exp_compatibility(self, manifold_fixture) -> None:
        """
        Checks that `exp` can handle broadcast-able `points` and `vectors`
        """
        parameters, manifold = manifold_fixture

        points = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 1, 1, 3, 3)
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
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        new_points = manifold.exp(points, vectors)

        assert new_points.shape == (5, 1, 4, 3, 3)
        assert torch.allclose(points, new_points, atol=parameters["tolerance"])

    def test_exp_correctness(self, manifold_fixture) -> None:
        """
        Checks that `exp` produce the correct results
        """
        parameters, manifold = manifold_fixture

        points = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 1, 3, 3)
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
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        new_points = manifold.exp(points, vectors)

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
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        for repetition in range(new_points.shape[0]):
            for distinct_point in range(new_points.shape[1]):
                assert torch.allclose(
                    new_points[repetition, distinct_point],
                    reference_points[distinct_point],
                    atol=parameters["tolerance"],
                )

    def test_log_compatibility(self, manifold_fixture) -> None:
        """
        Checks that `log` can handle broadcast-able `starts` and `ends`
        """
        parameters, manifold = manifold_fixture

        starts = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 1, 1, 3, 3)
        starts = starts.repeat(5, 2, 1, 1, 1)

        ends = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 3, 3)
        ends = ends.repeat(4, 1, 1)

        vectors = manifold.log(starts, ends)

        assert vectors.shape == (5, 2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(vectors, reference_vectors, atol=parameters["tolerance"])

    def test_log_correctness(self, manifold_fixture) -> None:
        """
        Checks that `log` produces correct results for small angles
        """
        parameters, manifold = manifold_fixture

        starts = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 1, 3, 3)
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
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        vectors = manifold.log(starts, ends)

        assert vectors.shape == (2, 4, 3)

        reference_vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        for repetition in range(vectors.shape[0]):
            for distinct_vector in range(vectors.shape[1]):
                assert torch.allclose(
                    vectors[repetition, distinct_vector],
                    reference_vectors[distinct_vector],
                    atol=parameters["tolerance"],
                )

    def test_exp_log_connection(self, manifold_fixture) -> None:
        """
        Checks that `log(points, exp(points, vectors)) = vectors`
        """
        parameters, manifold = manifold_fixture

        points = torch.eye(
            3, dtype=parameters["data_type"], device=parameters["device"]
        ).view(1, 1, 3, 3)
        points = points.repeat(2, 1, 1, 1)

        vectors = torch.tensor(
            [
                [0.25 * torch.pi, 0.0, 0.0],
                [0.0, 0.75 * torch.pi, 0.0],
                [0.0, 0.0, torch.pi / 3.0],
                [0.0, 0.0, -2.0 * torch.pi / 3.0],
            ],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        computed_vectors = manifold.log(points, manifold.exp(points, vectors))

        assert computed_vectors.shape == (2, 4, 3)

        for repetition in range(computed_vectors.shape[0]):
            for distinct_point in range(computed_vectors.shape[1]):
                assert torch.allclose(
                    computed_vectors[repetition, distinct_point],
                    vectors[distinct_point],
                    atol=parameters["tolerance"],
                )
