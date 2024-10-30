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
    manifold: manifolds.Manifold,
    time: torch.Tensor
):

    axis = axis_process.at(time = time).sample(num_samples = rotations.shape[0])
    angles = angle_process.at(time = time).sample(num_samples = rotations.shape[0])

    rotations = manifold.exp(

        points = rotations,

        tangent_vectors = torch.einsum(
            'ij..., ij... -> ij...',
            axis,
            angles
        )

    )

    time_tensor = time.view(time.numel(), 1)

    # WARN: mean square displacement seems manual and error prone
    angular_speeds = 0.6 * time_tensor ** 3 * angle_process.at(time = time).score_function(
        points = angles - torch.sin(angles)
    ) / (
        1.0 - torch.cos(angles)
    )

    tangent_velocities = torch.einsum(
        'ij..., ij... -> ij...',
        axis,
        angular_speeds
    )

    rotations = rotations.flatten(0, 1)
    rotations = rotations.flatten(-2)
    tangent_velocities = tangent_velocities.flatten(0, 1)

    return {
        'time': time,
        'rotations': rotations,
        'velocities': tangent_velocities
    }


def create_rotation_pipeline(
    num_sample_duplicates: int,
    num_timestamps: int,
    device: torch.device
) -> dataprocessing.Transform:

    orientation_process = stochasticprocesses.multivariate.UniformSphere(dimension = 3)
    orientation_process.to(device)

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

    time = 2.5 * torch.rand(size = (num_timestamps,), device = device)

    rotation_pipeline = dataprocessing.Pipeline(
        transforms = [

            # convert a NumPy array into a Torch tensor
            lambda rotations: torch.tensor(rotations, dtype = torch.float),

            # send tensor to the assigned computing device
            lambda rotations: rotations.to(device),

            # duplicate each data to add multiple noise simultaneously
            lambda rotations: rotations.unsqueeze(0).expand(
                num_sample_duplicates, *rotations.shape
            ).flatten(0, 1),

            # compute the random new points and how to return to the original points
            lambda rotations: (
                add_noise(
                    rotations = rotations,
                    axis_process = orientation_process,
                    angle_process = angle_process,
                    manifold = manifold,
                    time = time
                )
            )

        ]
    )

    return rotation_pipeline
