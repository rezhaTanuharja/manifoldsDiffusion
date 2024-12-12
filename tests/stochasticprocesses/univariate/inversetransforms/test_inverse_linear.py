import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    InverseTransform,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cdf.polynomials import (
    Linear,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)
from diffusionmodels.stochasticprocesses.univariate.uniform import UniformDensity


@pytest.fixture(scope="class")
def process_float():
    return InverseTransform(
        distribution=Linear(
            support={"lower": 0.0, "upper": 1.0}, data_type=torch.float32
        ),
        root_finder=Bisection(num_iterations=5),
    )


class TestOperationsFloat:
    def test_get_dimension(self, process_float):
        dimension = process_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_sample(self, process_float):
        samples = process_float.sample(num_samples=5)

        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (1, 5)
        assert samples.dtype == torch.float32

        assert torch.all((samples >= 0.0) & (samples <= 1.0))

        assert torch.std(samples) > 0

    def test_density(self, process_float):
        density = process_float.density

        assert isinstance(density, UniformDensity)


@pytest.fixture(scope="class")
def process_double():
    return InverseTransform(
        distribution=Linear(
            support={"lower": 0.0, "upper": 1.0}, data_type=torch.float64
        ),
        root_finder=Bisection(num_iterations=5),
        data_type=torch.float64,
    )


class TestOperationsdouble:
    def test_get_dimension(self, process_double):
        dimension = process_double.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_sample(self, process_double):
        samples = process_double.sample(num_samples=5)

        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (1, 5)
        assert samples.dtype == torch.float64

        assert torch.all((samples >= 0.0) & (samples <= 1.0))

        assert torch.std(samples) > 0

    def test_density(self, process_double):
        density = process_double.density

        assert isinstance(density, UniformDensity)
