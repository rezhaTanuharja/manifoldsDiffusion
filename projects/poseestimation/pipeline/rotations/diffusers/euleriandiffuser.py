"""
poseestimation.pipeline.rotations.diffusers.euleriandiffuser
============================================================

Add random noise to rotation matrices using a diffusion process
"""


import diffusionmodels.manifolds as manifolds
import diffusionmodels.dataprocessing as dataprocessing
import diffusionmodels.eulerian.stochasticprocesses as stochasticprocesses
import diffusionmodels.eulerian.stochasticprocesses.univariate as univariate
import torch

from typing import Tuple


class DiffusedAxisAngle(stochasticprocesses.StochasticProcess):

    def __init__(self) -> None:

        self._axis_process = stochasticprocesses.multivariate.UniformSphere(dimension = 3)

        angle_distribution_function = univariate.functions.periodic.HeatKernel(
            num_waves = 1000,
            mean_squared_displacement = lambda t: 0.15 * t ** 4
        )

        # for sampling: invert CDF using the bisection method
        inversion_method = univariate.inversion.Bisection(
            num_iterations = 12,
            pretransform = lambda points: points - torch.sin(points)
        )

        # angle distribution is defined by its CDF and sampled from using inverse transform
        self._angle_process = stochasticprocesses.univariate.InverseTransform(
            cumulative_distribution_function = angle_distribution_function,
            inversion_method = inversion_method
        )


    def dimension(self) -> Tuple[int, ...]:
        return self._axis_process.dimension()


    def to(self, device: torch.device) -> None:
        self._axis_process.to(device)
        self._angle_process.to(device)


    def at(self, time: torch.Tensor) -> stochasticprocesses.StochasticProcess:

        self._axis_process = self._axis_process.at(time = time)
        self._angle_process = self._angle_process.at(time = time)

        return self


    def density(self, points: torch.Tensor) -> torch.Tensor:

        return (
            self._axis_process.density(points)
            *
            self._angle_process.density(points)
        )

    
    def score_function(self, points: torch.Tensor) -> torch.Tensor:

        return(
            self._axis_process.score_function(points = points)
            +
            self._angle_process.score_function(points = points)
        )

    def sample(self, num_samples: int) -> torch.Tensor:

        axis_samples = self._axis_process.sample(num_samples = num_samples)
        angle_samples = self._angle_process.sample(num_samples = num_samples)

        return torch.einsum(
            'ij..., ij... -> ij...',
            axis_samples,
            angle_samples
        )


def create_rotation_pipeline(device: torch.device) -> dataprocessing.Pipeline:

    axis_angle_stochastic_process = DiffusedAxisAngle()
    axis_angle_stochastic_process.to(device)

    manifold = manifolds.SpecialOrthogonal3()

    rotation_pipeline = dataprocessing.Pipeline(
        transforms = [

            # convert a NumPy array into a Torch tensor
            lambda rotations: torch.tensor(rotations, dtype = torch.float),

            # send tensor to the assigned computing device
            lambda rotations: rotations.to(device),

            lambda rotations: rotations.unsqueeze(0).expand(
                3, *rotations.shape
            ).flatten(0, 1),

            lambda rotations: (
                manifold.exp(

                    rotation_matrices = rotations,

                    axis_angle_vectors = axis_angle_stochastic_process.at(
                        time = torch.Tensor([0.0, 0.2, 0.4])
                    ).sample(
                        num_samples = rotations.shape[0]
                    )

                )
            )

            #TODO: pipeline should return the diffused rotation and
            # the velocity to go back

        ]
    )

    return rotation_pipeline
