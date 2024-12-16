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
            size=(1, 64, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        values = uniform_distribution_float(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (1, 64)

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
            size=(1, 64, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_distribution_float.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1, 64)

        reference_values = torch.zeros(
            size=(1, 64), dtype=torch.float32, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )

    def test_change_time(self, uniform_distribution_float):
        time = torch.randn(size=(50,), dtype=torch.float32, device=torch.device("cpu"))

        distribution = uniform_distribution_float.at(time)

        points = torch.randn(
            size=(50, 64, 1), dtype=torch.float32, device=torch.device("cpu")
        )
        gradient = uniform_distribution_float.gradient(points)

        values = distribution(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (50, 64)

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
        assert gradient.shape == (50, 64)

        reference_values = torch.zeros(
            size=(50, 64), dtype=torch.float32, device=torch.device("cpu")
        )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )


@pytest.fixture(scope="class")
def one_wave_distribution_float():
    return PeriodicHeatKernel(num_waves=1, mean_squared_displacement=lambda time: time)


class TestOperationsOneWaveFloat:
    def test_get_dimension(self, one_wave_distribution_float):
        dimension = one_wave_distribution_float.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_call(self, one_wave_distribution_float):
        points = torch.randn(size=(1, 3, 1))
        values = one_wave_distribution_float(points)

        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32
        assert values.device == torch.device("cpu")
        assert values.shape == (1, 3)

        reference_values = 1.0 / torch.pi * torch.ones(size=(1, 3))

        for i in range(1):
            reference_values = reference_values + 2.0 / torch.pi * torch.cos(
                (i + 1) * points.squeeze(-1)
            )

        assert torch.allclose(values, reference_values, rtol=1e-7)

    def test_gradient_call(self, one_wave_distribution_float):
        points = torch.tensor(
            [0.0, torch.pi / 4.0, torch.pi / 6.0],
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).reshape(1, 3, 1)

        points = torch.randn(size=(1, 3, 1))

        gradient = one_wave_distribution_float.gradient(points)

        assert isinstance(gradient, torch.Tensor)
        assert gradient.dtype == torch.float32
        assert gradient.device == torch.device("cpu")
        assert gradient.shape == (1, 3)

        # reference_values = -1.0 / torch.pi * torch.tensor([0.0, 2**0.5, 1.0])

        reference_values = torch.zeros(size=(1, 3))

        for i in range(1):
            reference_values = reference_values - 2.0 / torch.pi * torch.sin(
                (i + 1) * points.squeeze(-1)
            )

        assert torch.allclose(
            gradient,
            reference_values,
            rtol=1e-16,
        )

    # def test_change_time(self, one_wave_distribution_float):
    #     # time = torch.tensor([0.0, 1.0])
    #     time = torch.randn(size=(2,))
    #
    #     distribution = one_wave_distribution_float.at(time)
    #
    #     # points = torch.tensor(
    #     #     [
    #     #         [0.0, torch.pi / 4.0, torch.pi / 3.0],
    #     #         [0.0, torch.pi / 3.0, torch.pi / 2.0],
    #     #     ],
    #     #     dtype=torch.float32,
    #     #     device=torch.device("cpu"),
    #     # ).reshape(2, 3, 1)
    #
    #     points = torch.randn(size=(2, 3, 1))
    #
    #     values = distribution(points)
    #
    #     assert isinstance(values, torch.Tensor)
    #     assert values.dtype == torch.float32
    #     assert values.device == torch.device("cpu")
    #     assert values.shape == (2, 3)
    #
    #     reference_values_first = torch.tensor(
    #         [3.0 / torch.pi, (1.0 + 2**0.5) / torch.pi, 2.0 / torch.pi],
    #     )
    #
    #     reference_values_second = 1.0 / torch.pi + torch.exp(
    #         torch.tensor(
    #             [
    #                 -1,
    #             ],
    #             dtype=torch.float32,
    #             device=torch.device("cpu"),
    #         )
    #     ) * torch.tensor(
    #         [2.0 / torch.pi, 1.0 / torch.pi, 0.0],
    #     )
    #
    #     reference_values = torch.vstack(
    #         [reference_values_first, reference_values_second]
    #     )
    #
    #     assert torch.allclose(values, reference_values, rtol=1e-7)
    #
    #     gradient = distribution.gradient(points)
    #
    #     assert isinstance(gradient, torch.Tensor)
    #     assert gradient.dtype == torch.float32
    #     assert gradient.device == torch.device("cpu")
    #     assert gradient.shape == (2, 3)
    #
    #     reference_values_first = torch.tensor(
    #         [
    #             0.0,
    #             -(2**0.5) / torch.pi,
    #             -(3**0.5) / torch.pi,
    #         ],
    #     )
    #
    #     reference_values_second = -torch.exp(
    #         torch.tensor(
    #             [
    #                 -1,
    #             ],
    #             dtype=torch.float32,
    #             device=torch.device("cpu"),
    #         )
    #     ) * torch.tensor(
    #         [0.0, 3**0.5 / torch.pi, 2.0 / torch.pi],
    #     )
    #
    #     reference_values = torch.vstack(
    #         [reference_values_first, reference_values_second]
    #     )
    #
    #     assert torch.allclose(
    #         gradient,
    #         reference_values,
    #         rtol=1e-7,
    #     )
