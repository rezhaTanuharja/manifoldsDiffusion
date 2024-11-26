"""
Provides unit test for the sequential dataprocessing module.

Author
------
Rezha Adrian Tanuharja

Date
----
2024-11-26
"""


import pytest
import jax
import jax.numpy as jnp

from ...dataprocessing import Transform, sequential


def test_construction() -> None:
    """
    Checks that `Pipeline` can be constructed as an instance of `Transform`
    """

    try:
        pipeline = sequential.Pipeline(
            transforms = [
                lambda data: data,
            ]
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    assert isinstance(pipeline, Transform)


def test_nested_construction() -> None:
    """
    Checks that a `Pipeline` can consist of two `Pipeline` instances
    """

    try:

        preprocess = sequential.Pipeline(
            transforms = [
                lambda data: data,
            ]
        )

        postprocess = sequential.Pipeline(
            transforms = [
                lambda data: data
            ]
        )

        pipeline = sequential.Pipeline(
            transforms = [
                preprocess,
                postprocess,
            ]
        )

    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    assert isinstance(pipeline, Transform)


@pytest.fixture(scope = "class")
def data_cpu():

    return jnp.array([
        [ 0.3,  0.1,  0.5],
        [ 1.5,  2.2, -0.5],
        [-0.3, -0.1,  0.5],
        [-2.3,  0.5,  0.1],
    ])


@pytest.mark.cpu
class TestCPUPipeline:
    """
    A group of `Pipeline` tests to be performed on CPU
    """

    def test_elementwise_operation(self, data_cpu) -> None:
        """
        Checks that `Pipeline` can perform element-wise transformations correctly
        """

        try:

            pipeline = sequential.Pipeline(
                transforms = [
                    lambda data: 0.5 + data,
                    lambda data: data ** 2,
                ]
            )

            output = pipeline(data_cpu)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert output.shape == (4, 3)

        reference_output = jnp.array([
            [0.64, 0.36, 1.00],
            [4.00, 7.29, 0.00],
            [0.04, 0.16, 1.00],
            [3.24, 1.00, 0.36]
        ])

        assert jnp.allclose(output, reference_output, atol = 1e-12)


    def test_nested_pipeline(self, data_cpu) -> None:
        """
        Checks that nested `Pipeline` still provides the correct results
        """

        try:

            preprocess = sequential.Pipeline(
                transforms = [
                    lambda data: jnp.sin(data),
                    lambda data: data ** 2,
                ]
            )

            postprocess = sequential.Pipeline(
                transforms = [
                    lambda data: data - jnp.mean(data, keepdims = True),
                    lambda data: data / (jnp.std(data, keepdims = True) + 1e-6),
                ]
            )

            pipeline = sequential.Pipeline(
                transforms = [
                    preprocess,
                    postprocess,
                ]
            )

            staggered_output = postprocess(preprocess(data_cpu))
            nested_output = pipeline(data_cpu)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert jnp.allclose(staggered_output, nested_output, atol = 1e-12)


@pytest.fixture(scope = "class")
def data_gpu():

    gpu = jax.devices("gpu")[0]

    data = jax.device_put(

        jnp.array([
            [ 0.3,  0.1,  0.5],
            [ 1.5,  2.2, -0.5],
            [-0.3, -0.1,  0.5],
            [-2.3,  0.5,  0.1],
        ]),

        device = gpu

    )

    return data


@pytest.mark.gpu
class TestGPUPipeline:
    """
    A group of `Pipeline` tests to be performed on GPU
    """

    def test_elementwise_operation(self, data_gpu) -> None:
        """
        Checks that `Pipeline` can perform element-wise transformations correctly
        """

        try:

            pipeline = sequential.Pipeline(
                transforms = [
                    lambda data: 0.5 + data,
                    lambda data: data ** 2,
                ]
            )

            output = pipeline(data_gpu)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert output.shape == (4, 3)

        reference_output = jnp.array([
            [0.64, 0.36, 1.00],
            [4.00, 7.29, 0.00],
            [0.04, 0.16, 1.00],
            [3.24, 1.00, 0.36]
        ])

        assert jnp.allclose(output, reference_output, atol = 1e-12)


    def test_nested_pipeline(self, data_gpu) -> None:
        """
        Checks that nested `Pipeline` still provides the correct results
        """

        try:

            preprocess = sequential.Pipeline(
                transforms = [
                    lambda data: jnp.sin(data),
                    lambda data: data ** 2,
                ]
            )

            postprocess = sequential.Pipeline(
                transforms = [
                    lambda data: data - jnp.mean(data, keepdims = True),
                    lambda data: data / (jnp.std(data, keepdims = True) + 1e-6),
                ]
            )

            pipeline = sequential.Pipeline(
                transforms = [
                    preprocess,
                    postprocess,
                ]
            )

            staggered_output = postprocess(preprocess(data_gpu))
            nested_output = pipeline(data_gpu)

        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        assert jnp.allclose(staggered_output, nested_output, atol = 1e-12)
