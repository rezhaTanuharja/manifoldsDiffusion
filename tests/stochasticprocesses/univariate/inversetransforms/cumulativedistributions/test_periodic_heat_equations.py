"""
Test file for CDF based on the solution to the heat equation in a periodic domain.
"""

from itertools import product

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.heatequations import (
    PeriodicHeatKernel,
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

devices = (
    torch.device("cpu"),
    torch.device("cuda", 0),
)

data_types_tolerances = zip((torch.float32, torch.float64), (5e-2, 5e-6))

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
def periodic_heat_kernel_fixture(request):
    parameters = request.param

    density = PeriodicHeatKernel(
        num_waves=parameters["num_waves"],
        mean_squared_displacement=parameters["mean_squared_displacement"],
        alpha=parameters["alpha"],
        data_type=parameters["data_type"],
    )

    density.to(parameters["device"])

    return parameters, density


class TestPeriodicHeatKernel:
    def test_get_dimension(self, periodic_heat_kernel_fixture):
        _, density = periodic_heat_kernel_fixture

        dimension = density.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, periodic_heat_kernel_fixture):
        parameters, density = periodic_heat_kernel_fixture

        points = torch.randn(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )
        values = density(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]
        assert values.shape == points.shape

        if parameters["num_waves"] == 0:
            reference_values = torch.full_like(
                values,
                fill_value=1.0 / torch.pi,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        else:
            reference_values = (
                1.0
                / torch.pi
                * torch.ones(
                    size=points.shape,
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
                ) * torch.cos(wave_number * points)

        assert torch.allclose(values, reference_values, atol=parameters["tolerance"])

    def test_gradient_call(self, periodic_heat_kernel_fixture):
        parameters, density = periodic_heat_kernel_fixture

        points = torch.randn(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )
        gradient = density.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == parameters["data_type"]
        assert gradient.device == parameters["device"]
        assert gradient.shape == points.shape

        if parameters["num_waves"] == 0:
            reference_values = torch.zeros(
                size=points.shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        else:
            reference_values = torch.zeros(
                size=points.shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
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
                )

        assert torch.allclose(
            gradient,
            reference_values,
            atol=parameters["tolerance"],
        )

    def test_change_time(self, periodic_heat_kernel_fixture):
        parameters, density = periodic_heat_kernel_fixture

        time = torch.abs(
            torch.randn(
                size=times_shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        density = density.at(time)

        points = torch.randn(
            size=points_shape,
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        values = density(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == parameters["data_type"]
        assert values.device == parameters["device"]
        assert values.shape == points.shape

        if parameters["num_waves"] == 0:
            reference_values = torch.full_like(
                values,
                fill_value=1.0 / torch.pi,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        else:
            reference_values = (
                1.0
                / torch.pi
                * torch.ones(
                    size=points.shape,
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

        gradient = density.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == parameters["data_type"]
        assert gradient.device == parameters["device"]
        assert gradient.shape == points.shape

        if parameters["num_waves"] == 0:
            reference_gradient = torch.zeros(
                size=points.shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        else:
            reference_gradient = torch.zeros(
                size=points.shape,
                dtype=parameters["data_type"],
                device=parameters["device"],
            )

            for wave_number in range(1, parameters["num_waves"] + 1):
                reference_gradient = (
                    reference_gradient
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

        assert torch.allclose(
            gradient,
            reference_gradient,
            atol=parameters["tolerance"],
        )
