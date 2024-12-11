import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import DensityFunction
from diffusionmodels.stochasticprocesses.univariate.uniform import Uniform


def test_construction() -> None:
    try:
        density_function = Uniform(support={"lower": 0.0, "upper": 1.0})
    except Exception as e:
        raise AssertionError(
            f"Manifold construction should not raise exception but got {e}"
        )

    assert isinstance(density_function, DensityFunction)


def test_get_dimension() -> None:
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
    def test_function_call(self, density_function_float) -> None:
        points = torch.rand(size=(3, 4, 5))

        try:
            density_values = density_function_float(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape

        assert torch.allclose(
            density_values,
            torch.full_like(input=points, fill_value=0.4, dtype=torch.float32),
        )

    def test_gradient_call(self, density_function_float) -> None:
        points = torch.rand(size=(3, 4, 5))

        try:
            gradient_values = density_function_float.gradient(points)
        except Exception as e:
            raise AssertionError(
                f"Calling gradient should not raise exception but got {e}"
            )

        assert gradient_values.shape == points.shape

        assert torch.allclose(
            gradient_values, torch.zeros_like(input=points, dtype=torch.float32)
        )


@pytest.fixture(scope="class")
def density_function_double():
    return Uniform(support={"lower": 1.5, "upper": 4.0}, data_type=torch.float64)


class TestOperationsDouble:
    def test_function_call(self, density_function_double) -> None:
        points = torch.rand(size=(3, 4, 5))

        try:
            density_values = density_function_double(points)
        except Exception as e:
            raise AssertionError(
                f"Calling density function should not raise exception but got {e}"
            )

        assert density_values.shape == points.shape

        assert torch.allclose(
            density_values,
            torch.full_like(input=points, fill_value=0.4, dtype=torch.float64),
        )

    def test_gradient_call(self, density_function_double) -> None:
        points = torch.rand(size=(3, 4, 5))

        try:
            gradient_values = density_function_double.gradient(points)
        except Exception as e:
            raise AssertionError(
                f"Calling gradient should not raise exception but got {e}"
            )

        assert gradient_values.shape == points.shape

        assert torch.allclose(
            gradient_values, torch.zeros_like(input=points, dtype=torch.float64)
        )
