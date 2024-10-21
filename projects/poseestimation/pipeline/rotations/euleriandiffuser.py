"""
poseestimation.pipeline.rotations.euleriandiffuser
==================================================

Add random noise to rotation matrices using a diffusion process
"""


import diffusionmodels.manifolds as manifolds
import diffusionmodels.dataprocessing as dataprocessing
import diffusionmodels.eulerian.stochasticprocesses as stochasticprocesses
import diffusionmodels.eulerian.stochasticprocesses.univariate as univariate
import torch


def add_noise(
    rotations: torch.Tensor,
    axis_process: stochasticprocesses.StochasticProcess,
    angle_process: stochasticprocesses.StochasticProcess,
    manifold: manifolds.Manifold
):

    time = torch.tensor([0.0, 0.2, 0.4])

    directions = axis_process.at(time = time).sample(num_samples = rotations.shape[0])
    distances = angle_process.at(time = time).sample(num_samples = rotations.shape[0])

    rotations = manifold.exp(

        points = rotations,

        tangent_vectors = torch.einsum(
            'ij..., ij... -> ij...',
            directions,
            distances
        )

    )

    #BUG: this speed is not scaled properly
    speeds = angle_process.score_function(points = distances)

    velocities = torch.einsum(
        'ij..., ij... -> ij...',
        directions,
        speeds
    )

    return {
        'time': time,
        'rotations': rotations,
        'velocities': velocities
    }


def create_rotation_pipeline(device: torch.device) -> dataprocessing.Pipeline:

    axis_process = stochasticprocesses.multivariate.UniformSphere(dimension = 3)
    axis_process.to(device)

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
    angle_process = stochasticprocesses.univariate.InverseTransform(
        cumulative_distribution_function = angle_distribution_function,
        inversion_method = inversion_method
    )

    angle_process.to(device)

    manifold = manifolds.SpecialOrthogonal3()
    manifold.to(device)

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
                add_noise(rotations, axis_process, angle_process, manifold)
            )

        ]
    )

    return rotation_pipeline
