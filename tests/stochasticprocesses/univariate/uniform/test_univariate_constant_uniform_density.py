"""
Checks that univariate uniform density function behaves as expected.
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.uniform import (
    ConstantUniformDensity,
)


@pytest.fixture(scope="class")
def density_function_float():
    return ConstantUniformDensity(support={"lower": 1.5, "upper": 4.0})


class TestOperationsFloat:
    """
    Checks correctness of math operations with float number
    """

    def test_get_dimension(self, density_function_float) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """

        dimension = density_function_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_function_call(self, density_function_float) -> None:
        """
        Checks that calling `UniformDensityDensity` as a function produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )

        density_values = density_function_float(points)

        assert density_values.shape == (1, *points.shape)
        assert density_values.dtype == torch.float32

        reference_values = torch.tensor(
            [
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

    def test_gradient_call(self, density_function_float) -> None:
        """
        Checks that calling the gradient function produces the right values
        """
        points = torch.rand(size=(3, 4, 5), dtype=torch.float32)

        gradient_values = density_function_float.gradient(points)

        assert gradient_values.shape == (1, *points.shape)
        assert gradient_values.dtype == torch.float32

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float32),
            rtol=1e-12,
        )

    def test_change_time(self, density_function_float):
        time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

        dimension = density_function_float.at(time).dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )

        density_values = density_function_float(points)

        assert density_values.shape == (3, *points.shape)
        assert density_values.dtype == torch.float32

        reference_values = torch.tensor(
            [
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

        points = torch.rand(size=(3, 4, 5), dtype=torch.float32)

        gradient_values = density_function_float.gradient(points)

        assert gradient_values.shape == (3, *points.shape)
        assert gradient_values.dtype == torch.float32

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float32),
            rtol=1e-12,
        )


@pytest.fixture(scope="class")
def density_function_double():
    return ConstantUniformDensity(
        support={"lower": 1.5, "upper": 4.0}, data_type=torch.float64
    )


class TestOperationsDouble:
    """
    Checks correctness of math operations with double precision
    """

    def test_function_call(self, density_function_double) -> None:
        """
        Checks that calling `UniformDensityDensity` as a function produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float64,
        )

        density_values = density_function_double(points)

        assert density_values.shape == (1, *points.shape)
        assert density_values.dtype == torch.float64

        reference_values = torch.tensor(
            [
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float64,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

    def test_gradient_call(self, density_function_double) -> None:
        """
        Checks that calling the gradient function produces the right values
        """
        points = torch.rand(size=(3, 4, 5), dtype=torch.float64)

        gradient_values = density_function_double.gradient(points)

        assert gradient_values.shape == (1, *points.shape)
        assert gradient_values.dtype == torch.float64

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float64),
            rtol=1e-12,
        )

    def test_change_time(self, density_function_double):
        time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        dimension = density_function_double.at(time).dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float64,
        )

        density_values = density_function_double(points)

        assert density_values.shape == (3, *points.shape)
        assert density_values.dtype == torch.float64

        reference_values = torch.tensor(
            [
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float64,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

        points = torch.rand(size=(3, 4, 5), dtype=torch.float64)

        gradient_values = density_function_double.gradient(points)

        assert gradient_values.shape == (3, *points.shape)
        assert gradient_values.dtype == torch.float64

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float64),
            rtol=1e-12,
        )
