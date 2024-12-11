"""
Checks that univariate uniform density function behaves as expected.
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import DensityFunction
from diffusionmodels.stochasticprocesses.univariate.uniform import Uniform


def test_construction() -> None:
    """
    Checks that `Uniform` is an instance of `DensityFunction`
    """
    try:
        density_function = Uniform(support={"lower": 0.0, "upper": 1.0})
    except Exception as e:
        raise AssertionError(
            f"Manifold construction should not raise exception but got {e}"
        )

    assert isinstance(density_function, DensityFunction)


def test_get_dimension() -> None:
    """
    Checks that dimension can be accessed and produces the correct values
    """
    try:
        density_function = Uniform(support={"lower": 0.0, "upper": 1.0})
        dimension = density_function.dimension
    except Exception as e:
        raise AssertionError(
            f"Accessing dimension should not raise exception but got {e}"
        )

    assert isinstance(dimension, tuple)
    assert len(dimension) == 1

    for entry in dimension:
        assert isinstance(entry, int)
        assert entry == 1


@pytest.fixture(scope="class")
def density_function_float():
    return Uniform(support={"lower": 1.5, "upper": 4.0}, data_type=torch.float32)


class TestOperationsFloat:
    """
    Checks correctness of math operations with float number
    """

    def test_function_call(self, density_function_float) -> None:
        """
        Checks that calling `Uniform` as a function produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )

        try:
            density_values = density_function_float(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape
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

        try:
            gradient_values = density_function_float.gradient(points)
        except Exception as e:
            raise AssertionError(
                f"Calling gradient should not raise exception but got {e}"
            )

        assert gradient_values.shape == points.shape
        assert gradient_values.dtype == torch.float32

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float32),
            rtol=1e-12,
        )


@pytest.fixture(scope="class")
def density_function_double():
    return Uniform(support={"lower": 1.5, "upper": 4.0}, data_type=torch.float64)


class TestOperationsDouble:
    """
    Checks correctness of math operations with double precision
    """

    def test_function_call(self, density_function_double) -> None:
        """
        Checks that calling `Uniform` as a function produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float64,
        )

        try:
            density_values = density_function_double(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape
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

        try:
            gradient_values = density_function_double.gradient(points)
        except Exception as e:
            raise AssertionError(
                f"Calling gradient should not raise exception but got {e}"
            )

        assert gradient_values.shape == points.shape
        assert gradient_values.dtype == torch.float64

        assert torch.allclose(
            gradient_values,
            torch.zeros_like(input=points, dtype=torch.float64),
            rtol=1e-12,
        )
