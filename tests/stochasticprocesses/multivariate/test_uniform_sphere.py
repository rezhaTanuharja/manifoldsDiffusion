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

dimensions = (1, 2, 4, 8)

test_parameters = [
    {
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "num_samples": num_samples,
        "device": device,
        "dimension": dimension,
    }
    for data_type_tolerance, num_samples, device, dimension in product(
        data_types_tolerances, nums_samples, devices, dimensions
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def uniform_sphere_process(request):
    parameters = request.param

    process = UniformSphere(
        dimension=parameters["dimension"], data_type=parameters["data_type"]
    )

    process.to(parameters["device"])

    return parameters, process


class TestOperations:
    """
    Checks correctness of math operations
    """

    def test_instance(
        self,
        uniform_sphere_process: Tuple[Dict[str, Any], StochasticProcess],
    ) -> None:
        """
        Checks that `UniformSphere` is an instance of `StochasticProcess`
        """
        _, process = uniform_sphere_process

        assert isinstance(process, StochasticProcess)

    def test_get_dimension(
        self,
        uniform_sphere_process: Tuple[Dict[str, Any], StochasticProcess],
    ) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        parameters, process = uniform_sphere_process

        dimension = process.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert (
                entry == parameters["dimension"]
            ), f"Entry value should be {parameters['dimension']}, got {entry} instead"

    def test_sample(
        self,
        uniform_sphere_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Checks that samples can be generated and produces the correct values
        """
        parameters, process = uniform_sphere_process

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
            parameters["dimension"],
        )

        norm = torch.norm(samples, dim=-1)

        assert torch.allclose(
            norm,
            torch.ones(
                size=norm.shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            ),
            atol=parameters["tolerance"],
        )

    def test_density(
        self,
        uniform_sphere_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Checks that density can be accessed and produces the correct result
        """
        parameters, process = uniform_sphere_process

        density = process.density

        assert isinstance(density, UniformSphereDensity)

        points = torch.randn(
            size=(parameters["num_samples"], parameters["dimension"]),
            dtype=parameters["data_type"],
            device=parameters["device"],
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
            / (2.0 * torch.pi ** (0.5 * (parameters["dimension"] + 1)))
            * torch.lgamma(torch.tensor(0.5 * (parameters["dimension"] + 1))).exp(),
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
        uniform_sphere_process: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Checks that density gradient can be accessed and produces the correct result
        """
        parameters, process = uniform_sphere_process

        density = process.density

        points = torch.randn(
            size=(parameters["num_samples"], parameters["dimension"]),
            dtype=parameters["data_type"],
            device=parameters["device"],
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

    def test_change_time(self, uniform_sphere_process):
        """
        Checks that everything still works after changing the internal time
        """
        parameters, process = uniform_sphere_process

        time = torch.abs(
            torch.rand(
                size=(256,), dtype=parameters["data_type"], device=parameters["device"]
            )
        )

        process = process.at(time)

        self.test_sample((parameters, process), time)
        self.test_density((parameters, process), time)
        self.test_density_gradient((parameters, process), time)
