"""
Checks that univariate uniform stochastic process behaves as expected.
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import StochasticProcess
from diffusionmodels.stochasticprocesses.univariate.uniform import (
    Uniform,
    UniformDensity,
)

points_shape = (1, 256)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)

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
def uniform_process(request):
    parameters = request.param

    process = Uniform(support=parameters["support"], data_type=parameters["data_type"])

    process.to(parameters["device"])

    return parameters, process


class TestOperations:
    """
    Checks correctness of math operations with float number
    """

    def test_get_dimension(
        self,
        uniform_process: Tuple[Dict[str, Any], StochasticProcess],
    ) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        _, process = uniform_process

        dimension = process.dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

    def test_sample(
        self,
        uniform_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Checks that samples can be generated and produces the correct values
        """
        parameters, process = uniform_process

        samples = process.sample(num_samples=parameters["num_samples"])

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == parameters["data_type"]

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert samples.shape == (
            *time.shape,
            parameters["num_samples"],
        )

        assert torch.all(
            (samples >= parameters["support"]["lower"])
            & (samples < parameters["support"]["upper"])
        )

        if parameters["num_samples"] > 1:
            assert torch.std(samples) > 0.0

    def test_gradient(
        self,
        uniform_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        parameters, process = uniform_process

        density = process.density

        assert isinstance(density, UniformDensity)

        lower = parameters["support"]["lower"]
        upper = parameters["support"]["upper"]

        scale = torch.rand(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        points = 0.5 * (lower + upper) + 0.5 * (upper - lower) * scale

        density_values = density(points)

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert density_values.shape == (*time.shape, *points.shape[1:])
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

    def test_hessian(
        self,
        uniform_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        parameters, process = uniform_process

        hessian = process.density.gradient

        lower = parameters["support"]["lower"]
        upper = parameters["support"]["upper"]

        scale = torch.rand(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        points = 0.5 * (lower + upper) + 0.5 * (upper - lower) * scale

        hessian_values = hessian(points)

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert hessian_values.shape == (*time.shape, *points.shape[1:])
        assert hessian_values.dtype == parameters["data_type"]
        assert hessian_values.device == parameters["device"]

        reference_values = torch.zeros(
            size=hessian_values.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(
            hessian_values,
            reference_values,
            atol=parameters["tolerance"],
        )

    def test_change_time(self, uniform_process):
        parameters, process = uniform_process

        time = torch.rand(
            size=(256,), dtype=parameters["data_type"], device=parameters["device"]
        )

        process = process.at(time)

        self.test_sample((parameters, process), time)
        self.test_gradient((parameters, process), time)
