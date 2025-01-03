"""
Checks the functionalities of uniform processes defined using the `InverseTransform`
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from diffusionmodels.stochasticprocesses.interfaces import StochasticProcess
from diffusionmodels.stochasticprocesses.univariate.inversetransforms import (
    InverseTransform,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.cumulativedistributions.polynomials import (
    ConstantLinear,
)
from diffusionmodels.stochasticprocesses.univariate.inversetransforms.rootfinders.bisection import (
    Bisection,
)
from diffusionmodels.stochasticprocesses.univariate.uniform import (
    UniformDensity,
)

nums_samples = (1, 5, 25, 125)
nums_iterations = (5, 10)

if torch.cuda.is_available():
    devices = (
        torch.device("cpu"),
        torch.device("cuda", 0),
    )
else:
    devices = (torch.device("cpu"),)

data_types_tolerances = zip((torch.float32, torch.float64), (1e-6, 1e-12))

supports = (
    {"lower": 2.3, "upper": 4.7},
    {"lower": -5.5, "upper": -1.8},
    {"lower": -1.2, "upper": 3.7},
)

test_parameters = [
    {
        "data_type": data_type_tolerance[0],
        "tolerance": data_type_tolerance[1],
        "support": support,
        "num_samples": num_samples,
        "device": device,
        "num_iterations": num_iterations,
    }
    for data_type_tolerance, support, num_samples, device, num_iterations in product(
        data_types_tolerances, supports, nums_samples, devices, nums_iterations
    )
]


@pytest.fixture(params=test_parameters, scope="class")
def process_fixture(request) -> Tuple[Dict[str, Any], StochasticProcess]:
    parameters = request.param

    process = InverseTransform(
        distribution=ConstantLinear(
            support=parameters["support"], data_type=parameters["data_type"]
        ),
        root_finder=Bisection(num_iterations=parameters["num_iterations"]),
        data_type=parameters["data_type"],
    )

    process.to(parameters["device"])

    return parameters, process


class TestOperationsFloat:
    def test_get_dimension(
        self,
        process_fixture: Tuple[Dict[str, Any], StochasticProcess],
    ) -> None:
        """
        Checks that the dimension can be accessed and the values are correct
        """
        _, process = process_fixture

        dimension = process.dimension

        assert isinstance(dimension, tuple)
        assert len(dimension) == 1

        for entry in dimension:
            assert isinstance(entry, int)
            assert entry == 1

    def test_sample(
        self,
        process_fixture: Tuple[Dict[str, Any], StochasticProcess],
        time: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Checks that samples can be generated and the values are correct
        """
        parameters, process = process_fixture

        samples = process.sample(num_samples=parameters["num_samples"])

        assert isinstance(samples, torch.Tensor)

        time = (
            time
            if time is not None
            else torch.zeros(
                size=(1,),
                dtype=parameters["data_type"],
                device=parameters["device"],
            )
        )

        assert samples.shape == (*time.shape, parameters["num_samples"])
        assert samples.dtype == parameters["data_type"]

        lower = parameters["support"]["lower"]
        upper = parameters["support"]["upper"]

        assert torch.all((samples >= lower) & (samples <= upper))

        if parameters["num_samples"] > 1:
            assert torch.std(samples) > 0

    def test_density(
        self, process_fixture: Tuple[Dict[str, Any], StochasticProcess]
    ) -> None:
        """
        Checks that density can be accessed and provides the correct values
        """
        _, process = process_fixture

        density = process.density

        assert isinstance(density, UniformDensity)

    def test_change_time(
        self, process_fixture: Tuple[Dict[str, Any], StochasticProcess]
    ) -> None:
        """
        Checks that everything is still correct after changing time
        """
        parameters, process = process_fixture

        time = torch.tensor(
            [0.0, 1.0, 2.0], dtype=parameters["data_type"], device=parameters["device"]
        )

        process = process.at(time)

        self.test_sample((parameters, process), time)
