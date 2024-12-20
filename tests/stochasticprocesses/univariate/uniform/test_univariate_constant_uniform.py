"""
Checks that univariate uniform stochastic process behaves as expected.
"""

from itertools import product

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.uniform import (
    ConstantUniform,
    ConstantUniformDensity,
)

points_shape = (1, 256)

devices = (
    torch.device("cpu"),
    torch.device("cuda", 0),
)

data_types_tolerances = zip((torch.float32, torch.float64), (1e-6, 1e-12))

supports = (
    {"lower": 2.3, "upper": 4.7},
    {"lower": -5.5, "upper": -1.8},
    {"lower": -1.2, "upper": 3.7},
)

nums_samples = (1, 5, 25, 125)

test_parameters = [
    {
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "support": support,
        "num_samples": num_samples,
        "device": device,
    }
    for data_type_tolerance, support, num_samples, device in product(
        data_types_tolerances, supports, nums_samples, devices
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def uniform_process_float(request):
    parameters = request.param

    distribution = ConstantUniform(
        support=parameters["support"], data_type=parameters["data_type"]
    )

    return parameters, distribution


class TestOperationsFloat:
    """
    Checks correctness of math operations with float number
    """

    def test_get_dimension(self, uniform_process_float) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        _, distribution = uniform_process_float

        dimension = distribution.dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

    def test_sample(self, uniform_process_float) -> None:
        """
        Checks that samples can be generated and produces the correct values
        """
        parameters, distribution = uniform_process_float

        samples = distribution.sample(num_samples=parameters["num_samples"])

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == parameters["data_type"]

        assert samples.shape == (
            1,
            parameters["num_samples"],
        )

        assert torch.all(
            (samples >= parameters["support"]["lower"])
            & (samples < parameters["support"]["upper"])
        )

        if parameters["num_samples"] > 1:
            assert torch.std(samples) > 0.0

    def test_gradient(self, uniform_process_float) -> None:
        parameters, distribution = uniform_process_float

        density = distribution.density

        assert isinstance(density, ConstantUniformDensity)

        lower = parameters["support"]["lower"]
        upper = parameters["support"]["upper"]

        scale = torch.rand(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        points = 0.5 * (lower + upper) + 0.5 * (upper - lower) * scale

        density_values = density(points)

        assert density_values.shape == points.shape
        assert density_values.dtype == parameters["data_type"]
        assert density_values.device == parameters["device"]

        reference_values = torch.where(
            torch.abs(scale) > 1.0,
            input=torch.zeros_like(
                points, dtype=parameters["data_type"], device=parameters["device"]
            ),
            other=1.0
            / (upper - lower)
            * torch.ones_like(
                points, dtype=parameters["data_type"], device=parameters["device"]
            ),
        )

        assert torch.allclose(
            density_values,
            reference_values,
            atol=parameters["tolerance"],
        )

    # def test_change_time(self, uniform_process_float):
    #     parameters, distribution = uniform_process_float
    #
    #     time = torch.tensor(
    #         [0.0, 1.0, 2.0], dtype=parameters["data_type"], device=parameters["device"]
    #     )
    #
    #     dimension = distribution.at(time).dimension
    #
    #     assert isinstance(dimension, tuple)
    #
    #     assert len(dimension) == 1
    #
    #     for entry in dimension:
    #         assert isinstance(entry, int)
    #         assert entry == 1, f"Entry value should be 1, got {entry} instead"
    #
    #     samples = uniform_process_float.sample(num_samples=50)
    #
    #     assert isinstance(samples, torch.Tensor)
    #     assert samples.dtype == torch.float32
    #
    #     assert samples.shape == (
    #         3,
    #         50,
    #         1,
    #     )
    #
    #     assert torch.all((samples >= 2.0) & (samples < 4.0))
    #     assert torch.std(samples) > 0.0
    #
    #     density = uniform_process_float.density
    #
    #     assert isinstance(density, ConstantUniformDensity)
    #


# @pytest.fixture(scope="class")
# def uniform_process_double():
#     return ConstantUniform(
#         support={"lower": 2.0, "upper": 4.0}, data_type=torch.float64
#     )
#
#
# class TestOperationsDouble:
#     """
#     Checks correctness of math operations with double number
#     """
#
#     def test_get_dimension(self, uniform_process_double) -> None:
#         """
#         Checks that dimension can be accessed and produces the correct values
#         """
#         dimension = uniform_process_double.dimension
#
#         assert isinstance(dimension, tuple)
#
#         assert len(dimension) == 1
#
#         for entry in dimension:
#             assert isinstance(entry, int)
#             assert entry == 1, f"Entry value should be 1, got {entry} instead"
#
#     def test_sample(self, uniform_process_double) -> None:
#         """
#         Checks that samples can be generated and produces the correct values
#         """
#         samples = uniform_process_double.sample(num_samples=50)
#
#         assert isinstance(samples, torch.Tensor)
#         assert samples.dtype == torch.float64
#
#         assert samples.shape == (
#             1,
#             50,
#             1,
#         )
#
#         assert torch.all((samples >= 2.0) & (samples < 4.0))
#         assert torch.std(samples) > 0.0
#
#     def test_gradient(self, uniform_process_double) -> None:
#         density = uniform_process_double.density
#
#         assert isinstance(density, ConstantUniformDensity)
#
#     def test_change_time(self, uniform_process_double):
#         time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
#
#         dimension = uniform_process_double.at(time).dimension
#
#         assert isinstance(dimension, tuple)
#
#         assert len(dimension) == 1
#
#         for entry in dimension:
#             assert isinstance(entry, int)
#             assert entry == 1, f"Entry value should be 1, got {entry} instead"
#
#         samples = uniform_process_double.sample(num_samples=50)
#
#         assert isinstance(samples, torch.Tensor)
#         assert samples.dtype == torch.float64
#
#         assert samples.shape == (
#             3,
#             50,
#             1,
#         )
#
#         assert torch.all((samples >= 2.0) & (samples < 4.0))
#         assert torch.std(samples) > 0.0
#
#         density = uniform_process_double.density
#
#         assert isinstance(density, ConstantUniformDensity)
