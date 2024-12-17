import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cdf.heatequations import (
    PeriodicHeatKernel,
)


@pytest.fixture(scope="class")
def uniform_distribution_float():
    return PeriodicHeatKernel(num_waves=0, mean_squared_displacement=lambda time: time)


class TestOperationsUniformFloat:
    def test_get_dimension(self, uniform_distribution_float):
        dimension = uniform_distribution_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, uniform_distribution_float):
        points = torch.randn(
            size=(2, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        values = uniform_distribution_float(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (2, 256)

        reference_values = torch.full_like(
            values,
            fill_value=1.0 / torch.pi,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert torch.allclose(values, reference_values, rtol=1e-16)

    #
    def test_gradient_call(self, uniform_distribution_float):
        points = torch.randn(
            size=(1, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_distribution_float.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1, 256)

        reference_values = torch.zeros(
            size=(1, 256), dtype=torch.float32, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )

    def test_change_time(self, uniform_distribution_float):
        time = torch.randn(size=(128,), dtype=torch.float32, device=torch.device("cpu"))

        distribution = uniform_distribution_float.at(time)

        points = torch.randn(
            size=(128, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_distribution_float.gradient(points)

        values = distribution(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = torch.full_like(
            values,
            fill_value=1.0 / torch.pi,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert torch.allclose(values, reference_values, rtol=1e-16)

        gradient = uniform_distribution_float.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (128, 256)

        reference_values = torch.zeros(
            size=(128, 256), dtype=torch.float32, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )


@pytest.fixture(scope="class")
def multiple_wave_distribution_float():
    return PeriodicHeatKernel(num_waves=64, mean_squared_displacement=lambda time: time)


class TestOperationsMultipleWaveFloat:
    def test_get_dimension(self, multiple_wave_distribution_float):
        dimension = multiple_wave_distribution_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, multiple_wave_distribution_float):
        points = torch.randn(size=(128, 256, 1))
        values = multiple_wave_distribution_float(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = 1.0 / torch.pi * torch.ones(size=(128, 256))

        for wave_number in range(1, 65):
            reference_values = reference_values + 2.0 / torch.pi * torch.cos(
                wave_number * points.squeeze(-1)
            )

        assert torch.allclose(values, reference_values, rtol=1e-1)

    def test_gradient_call(self, multiple_wave_distribution_float):
        points = torch.randn(size=(128, 256, 1))
        gradient = multiple_wave_distribution_float.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (128, 256)

        reference_values = torch.zeros(size=(128, 256))

        for wave_number in range(1, 65):
            reference_values = (
                reference_values
                - 2.0
                / torch.pi
                * wave_number
                * torch.sin(wave_number * points.squeeze(-1))
            )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-1,
        )

    def test_change_time(self, multiple_wave_distribution_float):
        time = torch.arange(16)
        distribution = multiple_wave_distribution_float.at(time)

        points = torch.randn(size=(16, 256, 1))
        values = distribution(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (16, 256)

        reference_values = 1.0 / torch.pi * torch.ones(size=(16, 256))

        for time_index in range(16):
            for wave_number in range(1, 65):
                reference_values[time_index] = reference_values[
                    time_index
                ] + 2.0 / torch.pi * torch.exp(
                    -time[time_index] * wave_number**2
                ) * torch.cos(wave_number * points[time_index].squeeze(-1))

        assert torch.allclose(values, reference_values, rtol=1e-1)

        gradient = distribution.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (16, 256)

        reference_values = torch.zeros(size=(16, 256))

        for time_index in range(16):
            for wave_number in range(1, 65):
                reference_values[time_index] = reference_values[
                    time_index
                ] - 2.0 / torch.pi * torch.exp(
                    -time[time_index] * wave_number**2
                ) * wave_number * torch.sin(
                    wave_number * points[time_index].squeeze(-1)
                )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-1,
        )
