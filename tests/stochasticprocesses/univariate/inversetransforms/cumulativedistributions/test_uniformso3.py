"""
Checks that ConstantLinear cumulativedistributions behaves as expected
"""

from itertools import product

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    CumulativeDistributionFunction,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.uniformso3 import (
    Angle,
)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)

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
def cdf_fixture(request):
    parameters = request.param
    distribution = Angle(data_type=parameters["data_type"])

    return parameters, distribution


class TestOperations:
    def test_instance(self, cdf_fixture):
        _, distribution = cdf_fixture

        assert isinstance(distribution, CumulativeDistributionFunction)

    def test_zero_function_call(self, cdf_fixture):
        parameters, distribution = cdf_fixture

        points = torch.zeros(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert points.shape == values.shape
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        reference_values = torch.zeros(
            size=values.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_ones_function_call(self, cdf_fixture):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.ones(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert points.shape == values.shape
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        reference_values = torch.ones(
            size=values.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_random_function_call(self, cdf_fixture):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert points.shape == values.shape
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        assert torch.all(values >= 0.0)
        assert torch.all(values <= 1.0)

        assert torch.std(values) > 0.0

    def test_monotonicity(self, cdf_fixture):
        parameters, distribution = cdf_fixture
        points1 = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        points2 = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )

        values1 = distribution(points1)
        values2 = distribution(points2)

        assert torch.equal(points1 > points2, values1 > values2)

    def test_function_values(self, cdf_fixture):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        reference_values = 0.5 - 0.5 * torch.cos(points)

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])
