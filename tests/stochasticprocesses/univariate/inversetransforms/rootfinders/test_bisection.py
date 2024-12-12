import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)


@pytest.fixture(scope="class")
def rootfinder():
    return Bisection(num_iterations=5)


class TestOperationsFloat:
    """
    Checks solving ability in float
    """

    def test_solve_linear(self, rootfinder):
        """
        Checks solving ability for a linear function
        """
        target_values = torch.tensor(
            [
                [-1.0, 2.3, 4.7, -2.1, 5.5],
                [2.1, 4.2, 3.8, 1.7, -3.5],
                [-3.7, -4.4, 2.2, 1.2, 0.8],
            ],
            dtype=torch.float32,
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
            input=target_values, other=solutions - 5.0, atol=20.0 / 2**4
        )

    def test_solve_quadratic_positive(self, rootfinder):
        """
        Checks solving ability for a quadratic function
        """
        target_values = torch.tensor(
            [
                [4.0, 11.0, 20.0, 31.0, 44.0],
            ],
            dtype=torch.float32,
        )

        solutions = rootfinder.solve(
            function=lambda points: points**2 - 5.0,
            target_values=target_values,
            interval=(0.0, 10.0),
        )

        assert isinstance(solutions, torch.Tensor)
        assert solutions.shape == target_values.shape

        assert torch.all((solutions >= 0.0) & (solutions <= 10.0))

        reference_solutions = torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0]])

        assert torch.allclose(
            input=solutions, other=reference_solutions, atol=10.0 / 2**5
        )

    def test_solve_sin(self, rootfinder):
        """
        Checks solving ability for a sine function
        """
        target_values = torch.tensor(
            [0.0, 0.5, 0.5 * 2.0**0.5, 0.5 * 3.0**0.5], dtype=torch.float32
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
            [[0.0, torch.pi / 6.0, torch.pi / 4.0, torch.pi / 3.0]]
        )

        assert torch.allclose(
            input=solutions, other=reference_solutions, atol=10.0 / 2**5
        )


class TestOperationsDouble:
    """
    Checks solving ability in double
    """

    def test_solve_linear(self, rootfinder):
        """
        Checks solving ability for a linear function
        """
        target_values = torch.tensor(
            [
                [-1.0, 2.3, 4.7, -2.1, 5.5],
                [2.1, 4.2, 3.8, 1.7, -3.5],
                [-3.7, -4.4, 2.2, 1.2, 0.8],
            ],
            dtype=torch.float64,
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
            input=target_values, other=solutions - 5.0, atol=20.0 / 2**4
        )

    def test_solve_quadratic_positive(self, rootfinder):
        """
        Checks solving ability for a quadratic function
        """
        target_values = torch.tensor(
            [
                [4.0, 11.0, 20.0, 31.0, 44.0],
            ],
            dtype=torch.float64,
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
            [[3.0, 4.0, 5.0, 6.0, 7.0]], dtype=torch.float64
        )

        assert torch.allclose(
            input=solutions, other=reference_solutions, atol=10.0 / 2**5
        )

    def test_solve_sin(self, rootfinder):
        """
        Checks solving ability for a sine function
        """
        target_values = torch.tensor(
            [0.0, 0.5, 0.5 * 2.0**0.5, 0.5 * 3.0**0.5], dtype=torch.float64
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
            [[0.0, torch.pi / 6.0, torch.pi / 4.0, torch.pi / 3.0]], dtype=torch.float64
        )

        assert torch.allclose(
            input=solutions, other=reference_solutions, atol=10.0 / 2**5
        )
