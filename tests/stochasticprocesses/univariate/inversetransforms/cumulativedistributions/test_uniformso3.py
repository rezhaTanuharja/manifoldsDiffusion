"""
Checks that ConstantLinear cumulativedistributions behaves as expected
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    CumulativeDistributionFunction,
    DensityFunction,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.uniformso3 import (
    Angle,
    AngleDensity,
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
def cdf_fixture(request) -> Tuple[Dict[str, Any], CumulativeDistributionFunction]:
    parameters = request.param

    distribution = Angle(data_type=parameters["data_type"])
    distribution.to(parameters["device"])

    return parameters, distribution


class TestOperations:
    def test_instance(
        self, cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction]
    ):
        _, distribution = cdf_fixture

        assert isinstance(distribution, CumulativeDistributionFunction)

    def test_left_edge_function_call(
        self,
        cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction],
        time: Optional[torch.Tensor] = None,
    ):
        parameters, distribution = cdf_fixture

        points = torch.zeros(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert values.shape == (*time.shape, *points.shape[1:])

        reference_values = torch.zeros(
            size=values.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_right_edge_function_call(
        self,
        cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction],
        time: Optional[torch.Tensor] = None,
    ):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.ones(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert values.shape == (*time.shape, *points.shape[1:])

        reference_values = torch.ones(
            size=values.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_random_function_call(
        self,
        cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction],
        time: Optional[torch.Tensor] = None,
    ):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert values.shape == (*time.shape, *points.shape[1:])

        assert torch.all(values >= 0.0)
        assert torch.all(values <= 1.0)

        assert torch.std(values) > 0.0

    def test_monotonicity(
        self, cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction]
    ):
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

    def test_function_values_correctness(
        self, cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction]
    ):
        parameters, distribution = cdf_fixture
        points = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = distribution(points)

        reference_values = 0.5 - 0.5 * torch.cos(points)

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_density(
        self,
        cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction],
        time: Optional[torch.Tensor] = None,
    ):
        parameters, distribution = cdf_fixture

        density = distribution.gradient

        assert isinstance(density, DensityFunction)
        assert isinstance(density, AngleDensity)

        points = torch.pi * torch.rand(
            size=(1, 12), dtype=parameters["data_type"], device=parameters["device"]
        )
        values = density(points)

        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert values.shape == (*time.shape, *points.shape[1:])

        reference_values = 0.5 * torch.sin(points)

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_change_time(
        self, cdf_fixture: Tuple[Dict[str, Any], CumulativeDistributionFunction]
    ):
        parameters, distribution = cdf_fixture

        time = torch.abs(
            torch.rand(
                size=(256,), dtype=parameters["data_type"], device=parameters["device"]
            )
        )

        distribution = distribution.at(time)

        self.test_left_edge_function_call((parameters, distribution), time)
        self.test_right_edge_function_call((parameters, distribution), time)
        self.test_random_function_call((parameters, distribution), time)
        self.test_density((parameters, distribution), time)
