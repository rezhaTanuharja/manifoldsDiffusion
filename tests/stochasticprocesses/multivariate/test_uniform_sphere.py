"""
Checks that uniform sphere stochastic process behaves as expected.
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import StochasticProcess
from diffusionmodels.stochasticprocesses.multivariate.uniform import (
    UniformSphere,
    UniformSphereDensity,
)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)
data_types_tolerances = zip((torch.float32, torch.float64), (1e-6, 1e-12))

nums_samples = (1, 5, 25, 125)

test_parameters = [
    {
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "num_samples": num_samples,
        "device": device,
    }
    for data_type_tolerance, num_samples, device in product(
        data_types_tolerances, nums_samples, devices
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def uniform_sphere_process_float(request):
    parameters = request.param

    process = UniformSphere(dimension=2, data_type=parameters["data_type"])

    process.to(parameters["device"])

    return parameters, process


class TestOperations:
    """
    Checks correctness of math operations
    """

    def test_get_dimension(
        self,
        uniform_sphere_process_float: Tuple[Dict[str, Any], StochasticProcess],
    ) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        _, process = uniform_sphere_process_float

        dimension = process.dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 2, f"Entry value should be 1, got {entry} instead"

    def test_density(
        self,
        uniform_sphere_process_float: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        parameters, process = uniform_sphere_process_float

        density = process.density

        assert isinstance(density, UniformSphereDensity)

        points = torch.randn(
            size=(32, 2), dtype=parameters["data_type"], device=parameters["device"]
        )

        points = points / torch.norm(points, keepdim=True)

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

        assert isinstance(density_values, torch.Tensor)
        assert density_values.shape == (*time.shape, points.shape[-2])
        assert density_values.dtype == parameters["data_type"]
        assert density_values.device == parameters["device"]

        reference_values = torch.full_like(
            density_values,
            fill_value=1.0
            / (2.0 * torch.pi ** (0.5 * (2 + 1)))
            * torch.lgamma(torch.tensor(0.5 * (2 + 1))).exp(),
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(
            density_values,
            reference_values,
            atol=parameters["tolerance"],
        )

    def test_density_gradient(
        self,
        uniform_sphere_process_float: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        parameters, process = uniform_sphere_process_float

        density = process.density

        points = torch.randn(
            size=(32, 2), dtype=parameters["data_type"], device=parameters["device"]
        )

        points = points / torch.norm(points, keepdim=True)

        density_gradient_values = density.gradient(points)

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert isinstance(density_gradient_values, torch.Tensor)
        assert density_gradient_values.shape == (*time.shape, *points.shape[-2:])
        assert density_gradient_values.dtype == parameters["data_type"]
        assert density_gradient_values.device == parameters["device"]

        reference_values = torch.zeros_like(
            density_gradient_values,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(
            density_gradient_values,
            reference_values,
            atol=parameters["tolerance"],
        )
