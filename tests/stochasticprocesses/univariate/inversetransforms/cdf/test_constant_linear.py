"""
Checks that ConstantLinear CDF behaves as expected
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cdf.polynomials import (
    ConstantLinear,
)
from diffusionmodels.stochasticprocesses.univariate.uniform import (
    ConstantUniformDensity,
)


@pytest.fixture(scope="class")
def cdf_float():
    return ConstantLinear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float32)


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
                [
                    [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                    [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
                ],
            ],
            dtype=torch.float32,
        )

        times = torch.tensor(
            [
                0.0,
            ],
            dtype=torch.float32,
        )

        cdf_values = cdf_float(points, times)

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
        Checks that calling the gradient produces an instance of `UniformDensity`
        """
        density_function = cdf_float.gradient

        assert isinstance(density_function, ConstantUniformDensity)


@pytest.fixture(scope="class")
def cdf_double():
    return ConstantLinear(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float64)


class TestOperationsdouble:
    """
    Check correctness of math operations in double precision
    """

    def test_function_call(self, cdf_double) -> None:
        """
        Checks that evaluating the cdf produces the correct results
        """
        points = torch.tensor(
            [
                [
                    [1.8, 2.0, 2.4, 2.8, 3.2, 3.3, 3.7, 3.9, 4.0],
                    [0.9, 0.7, 0.3, -0.2, 3.4, 5.3, 2.2, 2.1, 2.0],
                ],
            ],
            dtype=torch.float64,
        )

        times = torch.tensor(
            [
                0.0,
            ],
            dtype=torch.float64,
        )

        cdf_values = cdf_double(points, times)

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
        Checks that calling the gradient produces an instance of `UniformDensity`
        """
        density_function = cdf_double.gradient

        assert isinstance(density_function, ConstantUniformDensity)
