"""
Checks that Linear CDF behaves as expected
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import DensityFunction
from diffusionmodels.stochasticprocesses.univariate.cdf.polynomials import Linear
from diffusionmodels.stochasticprocesses.univariate.interfaces import (
    CumulativeDistributionFunction,
)


def test_construction() -> None:
    """
    Checks that `Linear` is an instance of `CumulativeDistributionFunction`
    """
    try:
        cdf = Linear(support={"lower": 0.0, "upper": 1.0})
    except Exception as e:
        raise AssertionError(
            f"Manifold construction should not raise exception but got {e}"
        )

    assert isinstance(cdf, CumulativeDistributionFunction)


@pytest.fixture(scope="class")
def cdf_float():
    return Linear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float32)


class TestOperationsFloat:
    """
    Check correctness of math operations in float precision
    """

    def test_function_call(self, cdf_float) -> None:
        """
        Checks that evaluating the cdf produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )
        try:
            cdf_values = cdf_float(points)
        except Exception as e:
            raise AssertionError(f"Calling cdf should not raise exception but got {e}")

        assert cdf_values.shape == points.shape
        assert cdf_values.dtype == torch.float32

        reference_values = torch.tensor(
            [
                [0.00, 0.00, 0.20, 0.40, 0.60, 0.65, 0.85, 0.95, 1.00],
                [0.00, 0.00, 0.00, 0.00, 0.70, 1.00, 0.10, 0.05, 0.00],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(cdf_values, reference_values, rtol=1e-6)

    def test_gradient(self, cdf_float) -> None:
        """
        Checks that calling the gradient produces an instance of `DensityFunction`
        """
        density_function = cdf_float.gradient

        assert isinstance(density_function, DensityFunction)


@pytest.fixture(scope="class")
def cdf_gradient_float():
    cdf = Linear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float32)
    return cdf.gradient


class TestGradientOperationFloat:
    def test_get_dimension(self, cdf_gradient_float) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        dimension = cdf_gradient_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_function_call(self, cdf_gradient_float) -> None:
        """
        Checks that calling the gradient function produces the right values
        """
        points = torch.tensor(
            [
                [1.7, 2.0, 2.4, 2.8, 4.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )

        try:
            density_values = cdf_gradient_float(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape
        assert density_values.dtype == torch.float32

        reference_values = torch.tensor(
            [
                [0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

    def test_gradient_call(self, cdf_gradient_float) -> None:
        """
        Checks that calling the hessian function produces the right values
        """
        points = torch.rand(size=(3, 4, 5), dtype=torch.float32)

        try:
            gradient_values = cdf_gradient_float.gradient(points)
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
def cdf_double():
    return Linear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float64)


class TestOperationsDouble:
    """
    Check correctness of math operations in double precision
    """

    def test_function_call(self, cdf_double) -> None:
        """
        Checks that evaluating the cdf produces the correct results
        """
        points = torch.tensor(
            [
                [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float64,
        )
        try:
            cdf_values = cdf_double(points)
        except Exception as e:
            raise AssertionError(f"Calling cdf should not raise exception but got {e}")

        assert cdf_values.shape == points.shape
        assert cdf_values.dtype == torch.float64

        reference_values = torch.tensor(
            [
                [0.00, 0.00, 0.20, 0.40, 0.60, 0.65, 0.85, 0.95, 1.00],
                [0.00, 0.00, 0.00, 0.00, 0.70, 1.00, 0.10, 0.05, 0.00],
            ],
            dtype=torch.float64,
        )

        assert torch.allclose(cdf_values, reference_values, rtol=1e-6)

    def test_gradient(self, cdf_double) -> None:
        """
        Checks that calling the gradient produces an instance of `DensityFunction`
        """
        density_function = cdf_double.gradient

        assert isinstance(density_function, DensityFunction)


@pytest.fixture(scope="class")
def cdf_gradient_double():
    cdf = Linear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float64)
    return cdf.gradient


class TestGradientOperationDouble:
    def test_get_dimension(self, cdf_gradient_double) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        dimension = cdf_gradient_double.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_function_call(self, cdf_gradient_double) -> None:
        """
        Checks that calling the gradient function produces the right values
        """
        points = torch.tensor(
            [
                [1.7, 2.0, 2.4, 2.8, 4.2, 3.3, 3.7, 3.9, 4.0],
                [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float64,
        )

        try:
            density_values = cdf_gradient_double(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape
        assert density_values.dtype == torch.float64

        reference_values = torch.tensor(
            [
                [0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        )

        assert torch.allclose(
            density_values,
            reference_values,
            rtol=1e-12,
        )

    def test_gradient_call(self, cdf_gradient_double) -> None:
        """
        Checks that calling the hessian function produces the right values
        """
        points = torch.rand(size=(3, 4, 5), dtype=torch.float64)

        try:
            gradient_values = cdf_gradient_double.gradient(points)
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
