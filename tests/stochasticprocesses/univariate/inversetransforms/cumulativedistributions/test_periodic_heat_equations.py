"""
Test file for CDF based on the solution to the heat equation in a periodic domain.
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.heatequations import (
    PeriodicCumulativeEnergy,
    PeriodicHeatKernel,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.interfaces import (
    CumulativeDistributionFunction,
)

points_shape = (1024, 256)
times_shape = (1024,)

num_waves = (0, 8, 64, 512)

mean_squared_displacements = (
    lambda time: time**0.5,
    lambda time: time,
    lambda time: time**2,
)

alphas = (0, 1, 2, 3)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)

data_types_tolerances = zip((torch.float32, torch.float64), (3e-2, 5e-6))

test_parameters = [
    {
        "num_waves": num_wave,
        "mean_squared_displacement": mean_squared_displacement,
        "alpha": alpha,
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "device": device,
    }
    for num_wave, mean_squared_displacement, alpha, data_type_tolerance, device in product(
        num_waves, mean_squared_displacements, alphas, data_types_tolerances, devices
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def periodic_cumulative_energy_fixture(request):
    parameters = request.param

    distribution = PeriodicCumulativeEnergy(
        num_waves=parameters["num_waves"],
        mean_squared_displacement=parameters["mean_squared_displacement"],
        alpha=parameters["alpha"],
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    return parameters, distribution


class TestPeriodicCumulativeEnergy:
    """
    Checks the functionalities of `PeriodicCumulativeEnergy`
    """

    def test_get_support(
        self,
        periodic_cumulative_energy_fixture: Tuple[
            Dict[str, Any], CumulativeDistributionFunction
        ],
    ):
        """
        Checks that support can be accessed and the value is correct
        """
        _, distribution = periodic_cumulative_energy_fixture

        support = distribution.support

        assert isinstance(support, dict)
        assert "lower" in support.keys()
        assert "upper" in support.keys()

    def test_call(
        self,
        periodic_cumulative_energy_fixture: Tuple[
            Dict[str, Any], CumulativeDistributionFunction
        ],
        time: Optional[torch.Tensor] = None,
    ):
        """
        Checks that the CDF is callable and yields the correct value
        """

        parameters, distribution = periodic_cumulative_energy_fixture

        points = torch.clip(
            torch.randn(
                size=points_shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            ),
            min=0.0,
            max=torch.pi,
        )
        values = distribution(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]
        assert values.shape == points.shape

        reference_values = points / torch.pi

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        for wave_number in range(1, parameters["num_waves"] + 1):
            reference_values = reference_values + 2.0 / torch.pi * torch.binomial(
                torch.tensor(
                    [
                        parameters["num_waves"],
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
            ) / torch.binomial(
                torch.tensor(
                    [
                        parameters["num_waves"] + parameters["alpha"],
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
            ) / wave_number * torch.sin(wave_number * points) * torch.exp(
                -parameters["mean_squared_displacement"](time.unsqueeze(-1))
                * wave_number**2
            )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_get_gradient(
        self,
        periodic_cumulative_energy_fixture: Tuple[
            Dict[str, Any], CumulativeDistributionFunction
        ],
        time: Optional[torch.Tensor] = None,
    ):
        """
        Checks that gradient can be accessed and called to yield the correct value
        """

        parameters, distribution = periodic_cumulative_energy_fixture

        gradient = distribution.gradient

        assert isinstance(gradient, PeriodicHeatKernel)

        points = torch.randn(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        values = gradient(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]
        assert values.shape == points.shape

        reference_values = torch.full_like(
            values,
            fill_value=1.0 / torch.pi,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        for wave_number in range(1, parameters["num_waves"] + 1):
            reference_values = reference_values + 2.0 / torch.pi * torch.binomial(
                torch.tensor(
                    [
                        parameters["num_waves"],
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
            ) / torch.binomial(
                torch.tensor(
                    [
                        parameters["num_waves"] + parameters["alpha"],
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=parameters["data_type"],
                    device=parameters["device"],
                ),
            ) * torch.cos(wave_number * points) * torch.exp(
                -parameters["mean_squared_displacement"](time.unsqueeze(-1))
                * wave_number**2
            )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_get_hessian(
        self,
        periodic_cumulative_energy_fixture: Tuple[
            Dict[str, Any], CumulativeDistributionFunction
        ],
        time: Optional[torch.Tensor] = None,
    ):
        """
        Checks that hessian can be accessed and called to yield the correct value
        """

        parameters, distribution = periodic_cumulative_energy_fixture

        hessian = distribution.gradient.gradient

        points = torch.randn(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        values = hessian(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]
        assert values.shape == points.shape

        reference_values = torch.zeros(
            size=points.shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        for wave_number in range(1, parameters["num_waves"] + 1):
            reference_values = (
                reference_values
                - 2.0
                / torch.pi
                * wave_number
                * torch.binomial(
                    torch.tensor(
                        [
                            parameters["num_waves"],
                        ],
                        dtype=parameters["data_type"],
                        device=parameters["device"],
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=parameters["data_type"],
                        device=parameters["device"],
                    ),
                )
                / torch.binomial(
                    torch.tensor(
                        [
                            parameters["num_waves"] + parameters["alpha"],
                        ],
                        dtype=parameters["data_type"],
                        device=parameters["device"],
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=parameters["data_type"],
                        device=parameters["device"],
                    ),
                )
                * torch.sin(wave_number * points)
                * torch.exp(
                    -parameters["mean_squared_displacement"](time.unsqueeze(-1))
                    * wave_number**2
                )
            )

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_change_time(
        self,
        periodic_cumulative_energy_fixture: Tuple[
            Dict[str, Any], CumulativeDistributionFunction
        ],
    ):
        """
        Checks that all tests still pass after changing time
        """

        parameters, distribution = periodic_cumulative_energy_fixture

        time = torch.abs(
            torch.randn(
                size=times_shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        distribution = distribution.at(time)

        self.test_call((parameters, distribution), time)
        self.test_get_gradient((parameters, distribution), time)
        self.test_get_hessian((parameters, distribution), time)
