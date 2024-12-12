"""
Checks that univariate uniform stochastic process behaves as expected.
"""

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import StochasticProcess
from diffusionmodels.stochasticprocesses.univariate.uniform import Uniform


@pytest.fixture(scope="module")
def uniform_process_float():
    try:
        process = Uniform(support={"lower": 2.0, "upper": 4.0})
    except Exception as e:
        print(f"Fixture raised an exception: {e}")
        return None

    return process


def test_construction(uniform_process_float):
    assert isinstance(uniform_process_float, StochasticProcess)


@pytest.mark.skipif(
    not isinstance(uniform_process_float, StochasticProcess),
    reason="Failed to construct a uniform process",
)
class TestOperationsFloat:
    def test_get_dimension(self, uniform_process_float) -> None:
        try:
            dimension = uniform_process_float.dimension
        except Exception as e:
            assert False, f"Got an exception when accessing dimension: {e}"

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_sample(self, uniform_process_float) -> None:
        try:
            samples = uniform_process_float.sample(num_samples=50)
        except Exception as e:
            assert False, f"Got an exception when accessing dimension: {e}"

        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (50, 1)

        assert torch.all((samples >= 2.0) & (samples < 4.0))

        assert torch.std(samples) > 0.0
