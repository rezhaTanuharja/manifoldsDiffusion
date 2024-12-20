from itertools import product
from typing import Any, Dict, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.interfaces import (
    RootFinder,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)

nums_iterations = (5, 10)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)

data_types = (torch.float32, torch.float64)

test_parameters = [
    {
        "data_type": data_type,
        "device": device,
        "num_iterations": num_iterations,
    }
    for data_type, device, num_iterations in product(
        data_types, devices, nums_iterations
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def rootfinder_fixture(request):
    parameters = request.param

    rootfinder = Bisection(num_iterations=parameters["num_iterations"])

    return parameters, rootfinder


class TestOperationsFloat:
    """
    Checks solving ability in float
    """

    def test_solve_linear(self, rootfinder_fixture: Tuple[Dict[str, Any], RootFinder]):
        """
        Checks solving ability for a linear function
        """
        parameters, rootfinder = rootfinder_fixture

        target_values = torch.tensor(
            [
                [-1.0, 2.3, 4.7, -2.1, -8.5],
                [2.1, 4.2, 3.8, 1.7, -3.5],
                [-12.7, -4.4, 2.2, 1.2, 0.8],
            ],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        solutions = rootfinder.solve(
            function=lambda points: points - 5.0,
            target_values=target_values,
            interval=(-10.0, 10.0),
        )

        assert isinstance(solutions, torch.Tensor)
        assert solutions.shape == target_values.shape

        assert torch.all((solutions >= -10.0) & (solutions <= 10.0))

        assert torch.allclose(
            input=target_values,
            other=solutions - 5.0,
            atol=20.0 / 2 ** parameters["num_iterations"],
        )

    def test_solve_quadratic_positive(
        self, rootfinder_fixture: Tuple[Dict[str, Any], RootFinder]
    ):
        """
        Checks solving ability for a quadratic function
        """
        parameters, rootfinder = rootfinder_fixture

        target_values = torch.tensor(
            [
                [4.0, 11.0, 20.0, 31.0, 44.0],
            ],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        solutions = rootfinder.solve(
            function=lambda points: points**2 - 5.0,
            target_values=target_values,
            interval=(0.0, 10.0),
        )

        assert isinstance(solutions, torch.Tensor)
        assert solutions.shape == target_values.shape

        assert torch.all((solutions >= 0.0) & (solutions <= 10.0))

        reference_solutions = torch.tensor(
            [[3.0, 4.0, 5.0, 6.0, 7.0]],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(
            input=solutions,
            other=reference_solutions,
            atol=10.0 / 2 ** parameters["num_iterations"],
        )

    def test_solve_sin(self, rootfinder_fixture: Tuple[Dict[str, Any], RootFinder]):
        """
        Checks solving ability for a sine function
        """
        parameters, rootfinder = rootfinder_fixture

        target_values = torch.tensor(
            [0.0, 0.5, 0.5 * 2.0**0.5, 0.5 * 3.0**0.5],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        solutions = rootfinder.solve(
            function=lambda points: torch.sin(points),
            target_values=target_values,
            interval=(0.0, 0.5 * torch.pi),
        )

        assert isinstance(solutions, torch.Tensor)
        assert solutions.shape == target_values.shape

        assert torch.all((solutions >= 0.0) & (solutions <= 0.5 * torch.pi))

        reference_solutions = torch.tensor(
            [[0.0, torch.pi / 6.0, torch.pi / 4.0, torch.pi / 3.0]],
            dtype=parameters["data_type"],
            device=parameters["device"],
        )

        assert torch.allclose(
            input=solutions,
            other=reference_solutions,
            atol=10.0 / 2 ** parameters["num_iterations"],
        )
