import pytest
import torch

from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cdf.cesarosums import (
    CesaroSum,
    CesaroSumDensity,
)


@pytest.fixture(scope="class")
def uniform_density_float_cpu():
    return CesaroSumDensity(num_waves=0, mean_squared_displacement=lambda time: time)


class TestOperationsUniformDensityFloatCPU:
    def test_get_dimension(self, uniform_density_float_cpu):
        dimension = uniform_density_float_cpu.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, uniform_density_float_cpu):
        points = torch.randn(
            size=(2, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        values = uniform_density_float_cpu(points)

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

    def test_gradient_call(self, uniform_density_float_cpu):
        points = torch.randn(
            size=(1, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_density_float_cpu.gradient(points)

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

    def test_change_time(self, uniform_density_float_cpu):
        time = torch.randn(size=(128,), dtype=torch.float32, device=torch.device("cpu"))

        density = uniform_density_float_cpu.at(time)

        points = torch.randn(
            size=(128, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_density_float_cpu.gradient(points)

        values = density(points)

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

        gradient = uniform_density_float_cpu.gradient(points)

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
def multiple_wave_density_float_cpu():
    return CesaroSumDensity(num_waves=64, mean_squared_displacement=lambda time: time)


class TestOperationsMultipleWaveFloatCPU:
    def test_get_dimension(self, multiple_wave_density_float_cpu):
        dimension = multiple_wave_density_float_cpu.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, multiple_wave_density_float_cpu):
        points = torch.randn(
            size=(1024, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        values = multiple_wave_density_float_cpu(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (1024, 256)

        reference_values = 1.0 / torch.pi * torch.ones(size=(1024, 256))

        for wave_number in range(1, 65):
            reference_values = reference_values + 2.0 / torch.pi * torch.binomial(
                torch.tensor(
                    [
                        64 - 2,
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=torch.float32,
                ),
            ) / torch.binomial(
                torch.tensor(
                    [
                        64,
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=torch.float32,
                ),
            ) * torch.cos(wave_number * points.squeeze(-1))

        assert torch.allclose(values, reference_values, atol=1e-3)

    def test_gradient(self, multiple_wave_density_float_cpu):
        points = torch.randn(
            size=(1024, 256, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = multiple_wave_density_float_cpu.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1024, 256)

        reference_gradient = torch.zeros(
            size=(1024, 256), dtype=torch.float32, device=torch.device("cpu", 0)
        )

        for wave_number in range(1, 65):
            reference_gradient = (
                reference_gradient
                - 2.0
                / torch.pi
                * wave_number
                * torch.binomial(
                    torch.tensor(
                        [
                            64 - 2,
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=torch.float32,
                    ),
                )
                / torch.binomial(
                    torch.tensor(
                        [
                            64,
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=torch.float32,
                    ),
                )
                * torch.sin(wave_number * points.squeeze(-1))
            )

        assert torch.allclose(gradient, reference_gradient, atol=1e-3)


@pytest.fixture(scope="class")
def uniform_density_double_cpu():
    return CesaroSumDensity(num_waves=0, mean_squared_displacement=lambda time: time)


class TestOperationsUniformDensitydoubleCPU:
    def test_get_dimension(self, uniform_density_double_cpu):
        dimension = uniform_density_double_cpu.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, uniform_density_double_cpu):
        points = torch.randn(
            size=(2, 256, 1), dtype=torch.float64, device=torch.device("cpu")
        )
        values = uniform_density_double_cpu(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float64
        assert values.device == torch.device("cpu")
        assert values.shape == (2, 256)

        reference_values = torch.full_like(
            values,
            fill_value=1.0 / torch.pi,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        assert torch.allclose(values, reference_values, rtol=1e-16)

    def test_gradient_call(self, uniform_density_double_cpu):
        points = torch.randn(
            size=(1, 256, 1), dtype=torch.float64, device=torch.device("cpu")
        )
        gradient = uniform_density_double_cpu.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float64
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1, 256)

        reference_values = torch.zeros(
            size=(1, 256), dtype=torch.float64, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )

    def test_change_time(self, uniform_density_double_cpu):
        time = torch.randn(size=(128,), dtype=torch.float64, device=torch.device("cpu"))

        density = uniform_density_double_cpu.at(time)

        points = torch.randn(
            size=(128, 256, 1), dtype=torch.float64, device=torch.device("cpu")
        )
        gradient = uniform_density_double_cpu.gradient(points)

        values = density(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float64
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = torch.full_like(
            values,
            fill_value=1.0 / torch.pi,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        assert torch.allclose(values, reference_values, rtol=1e-16)

        gradient = uniform_density_double_cpu.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float64
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (128, 256)

        reference_values = torch.zeros(
            size=(128, 256), dtype=torch.float64, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )


@pytest.fixture(scope="class")
def multiple_wave_density_double_cpu():
    return CesaroSumDensity(num_waves=64, mean_squared_displacement=lambda time: time)


class TestOperationsMultipleWaveDoubleCPU:
    def test_get_dimension(self, multiple_wave_density_double_cpu):
        dimension = multiple_wave_density_double_cpu.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, multiple_wave_density_double_cpu):
        points = torch.randn(
            size=(1024, 256, 1), dtype=torch.float64, device=torch.device("cpu")
        )
        values = multiple_wave_density_double_cpu(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float64
        assert values.device == torch.device("cpu")
        assert values.shape == (1024, 256)

        reference_values = 1.0 / torch.pi * torch.ones(size=(1024, 256))

        for wave_number in range(1, 65):
            reference_values = reference_values + 2.0 / torch.pi * torch.binomial(
                torch.tensor(
                    [
                        64 - 2,
                    ],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=torch.float64,
                ),
            ) / torch.binomial(
                torch.tensor(
                    [
                        64,
                    ],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [
                        wave_number,
                    ],
                    dtype=torch.float64,
                ),
            ) * torch.cos(wave_number * points.squeeze(-1))

        assert torch.allclose(values, reference_values, atol=1e-6)

    def test_gradient(self, multiple_wave_density_double_cpu):
        points = torch.randn(
            size=(1024, 256, 1), dtype=torch.float64, device=torch.device("cpu")
        )
        gradient = multiple_wave_density_double_cpu.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float64
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1024, 256)

        reference_gradient = torch.zeros(
            size=(1024, 256), dtype=torch.float64, device=torch.device("cpu", 0)
        )

        for wave_number in range(1, 65):
            reference_gradient = (
                reference_gradient
                - 2.0
                / torch.pi
                * wave_number
                * torch.binomial(
                    torch.tensor(
                        [
                            64 - 2,
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=torch.float64,
                    ),
                )
                / torch.binomial(
                    torch.tensor(
                        [
                            64,
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor(
                        [
                            wave_number,
                        ],
                        dtype=torch.float64,
                    ),
                )
                * torch.sin(wave_number * points.squeeze(-1))
            )

        assert torch.allclose(gradient, reference_gradient, atol=1e-6)


@pytest.fixture(scope="class")
def uniform_distribution_float_cpu():
    return CesaroSum(num_waves=0, mean_squared_displacement=lambda time: time)


class TestOperationsUniformFloatCPU:
    def test_get_support(self, uniform_distribution_float_cpu):
        support = uniform_distribution_float_cpu.support

        assert isinstance(support, dict)
        assert len(support) == 2

        assert "lower" in support.keys()
        assert "upper" in support.keys()

        assert support["lower"] == 0.0
        assert support["upper"] == torch.pi

    def test_call(self, uniform_distribution_float_cpu):
        times = torch.randn(size=(128,))

        points = torch.zeros(size=(128, 256, 1), dtype=torch.float32, device="cpu")
        values = uniform_distribution_float_cpu(points, times)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = torch.zeros(size=(128, 256))

        assert torch.allclose(values, reference_values, rtol=1e-16)

        points = torch.pi * torch.ones(
            size=(128, 256, 1), dtype=torch.float32, device="cpu"
        )
        values = uniform_distribution_float_cpu(points, times)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = torch.ones(size=(128, 256))

        assert torch.allclose(values, reference_values, rtol=1e-16)

        points = torch.clip(
            torch.randn(
                size=(128, 256, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            min=0.0,
            max=torch.pi,
        )
        values = uniform_distribution_float_cpu(points, times)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (128, 256)

        reference_values = points.squeeze(-1) / torch.pi

        assert torch.allclose(values, reference_values, rtol=1e-16)

    def test_gradient(self, uniform_distribution_float_cpu):
        gradient = uniform_distribution_float_cpu.gradient

        assert isinstance(gradient, CesaroSumDensity)
