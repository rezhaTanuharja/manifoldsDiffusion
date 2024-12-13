"""
Checks that univariate uniform stochastic process behaves as expected.
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.uniform import (
    Uniform,
    UniformDensity,
)


@pytest.fixture(scope="class")
def uniform_process_float():
    return Uniform(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float32)


class TestOperationsFloat:
    """
    Checks correctness of math operations with float number
    """

    def test_get_dimension(self, uniform_process_float) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        dimension = uniform_process_float.dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

    def test_sample(self, uniform_process_float) -> None:
        """
        Checks that samples can be generated and produces the correct values
        """
        samples = uniform_process_float.sample(num_samples=50)

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == torch.float32

        assert samples.shape == (
            1,
            50,
            1,
        )

        assert torch.all((samples >= 2.0) & (samples < 4.0))
        assert torch.std(samples) > 0.0

    def test_gradient(self, uniform_process_float) -> None:
        density = uniform_process_float.density

        assert isinstance(density, UniformDensity)

    def test_change_time(self, uniform_process_float):
        time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

        dimension = uniform_process_float.at(time).dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

        samples = uniform_process_float.sample(num_samples=50)

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == torch.float32

        assert samples.shape == (
            3,
            50,
            1,
        )

        assert torch.all((samples >= 2.0) & (samples < 4.0))
        assert torch.std(samples) > 0.0

        density = uniform_process_float.density

        assert isinstance(density, UniformDensity)


@pytest.fixture(scope="class")
def uniform_process_double():
    return Uniform(support={"lower": 2.0, "upper": 4.0}, data_type=torch.float64)


class TestOperationsDouble:
    """
    Checks correctness of math operations with double number
    """

    def test_get_dimension(self, uniform_process_double) -> None:
        """
        Checks that dimension can be accessed and produces the correct values
        """
        dimension = uniform_process_double.dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

    def test_sample(self, uniform_process_double) -> None:
        """
        Checks that samples can be generated and produces the correct values
        """
        samples = uniform_process_double.sample(num_samples=50)

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == torch.float64

        assert samples.shape == (
            1,
            50,
            1,
        )

        assert torch.all((samples >= 2.0) & (samples < 4.0))
        assert torch.std(samples) > 0.0

    def test_gradient(self, uniform_process_double) -> None:
        density = uniform_process_double.density

        assert isinstance(density, UniformDensity)

    def test_change_time(self, uniform_process_double):
        time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        dimension = uniform_process_double.at(time).dimension

        assert isinstance(dimension, tuple)

        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1, f"Entry value should be 1, got {entry} instead"

        samples = uniform_process_double.sample(num_samples=50)

        assert isinstance(samples, torch.Tensor)
        assert samples.dtype == torch.float64

        assert samples.shape == (
            3,
            50,
            1,
        )

        assert torch.all((samples >= 2.0) & (samples < 4.0))
        assert torch.std(samples) > 0.0

        density = uniform_process_double.density

        assert isinstance(density, UniformDensity)
