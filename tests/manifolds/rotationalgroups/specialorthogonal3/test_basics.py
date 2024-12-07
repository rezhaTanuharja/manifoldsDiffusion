"""
Checks basic functionalities of the SpecialOrthogonal3 class.
"""

from diffusionmodels.manifolds import Manifold
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3


def test_construction() -> None:
    """
    Checks that `SpecialOrthogonal3` can be constructed as an instance of `Manifold`
    """

    try:
        manifold = SpecialOrthogonal3()

    except Exception as e:
        raise AssertionError(
            f"Manifold construction should not raise exception but got {e}"
        )

    assert isinstance(manifold, Manifold)


def test_get_dimension() -> None:
    """
    Checks that `dimension` can be accessed and the value is a tuple `(3, 3)`
    """

    try:
        manifold = SpecialOrthogonal3()
        dimension = manifold.dimension
    except Exception as e:
        raise AssertionError(
            f"Accessing dimension should not raise exception but got {e}"
        )

    assert isinstance(dimension, tuple)
    assert len(dimension) == 2

    for entry in dimension:
        assert isinstance(entry, int)
        assert entry == 3


def test_get_tangent_dimension() -> None:
    """
    Checks that `tangent_dimension` can be invoked and return the tuple `(3,)`
    """

    try:
        manifold = SpecialOrthogonal3()
        tangent_dimension = manifold.tangent_dimension
    except Exception as e:
        raise AssertionError(
            f"Accessing tangent dimension should not raise exception but got {e}"
        )

    assert isinstance(tangent_dimension, tuple)
    assert len(tangent_dimension) == 1

    for entry in tangent_dimension:
        assert isinstance(entry, int)
        assert entry == 3
