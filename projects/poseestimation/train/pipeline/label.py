from diffusionmodels.eulerian import stochasticprocesses as sp
from diffusionmodels import manifolds

import torch


manifold = manifolds.SpecialOrthogonal3()

inversion_method = sp.univariate.inversion.Bisection(
    num_iterations = 12,
    pretransform = lambda points: points - torch.sin(points)
)

cumulative_distribution_function = sp.univariate.functions.periodic.HeatKernel(
    num_waves = 1000,
    mean_squared_displacement = lambda t: 0.15 * t ** 4
)

angle_distribution = sp.univariate.InverseTransform(

    cumulative_distribution_function = cumulative_distribution_function,
    inversion_method = inversion_method

)

axis_distribution = sp.multivariate.UniformSphere(dimension = 3)


for obj in [
    manifold,
    cumulative_distribution_function,
    inversion_method,
    angle_distribution,
    axis_distribution,
]:
    obj = obj.to(torch.device('cuda'))


def noise(initial_point: torch.Tensor, time: torch.Tensor, num_samples: int):

    random_angle = angle_distribution.at(time = time).sample(num_samples = num_samples)
    random_axis = axis_distribution.at(time = time).sample(num_samples = num_samples)

    random_angle = random_angle.flatten(0, 1)
    random_axis = random_axis.flatten(0, 1)
    
    return {
        'location': manifold.exp(
            initial_point,
            torch.einsum(
                'i..., i... -> i...',
                random_angle,
                random_axis
            )
        ),
        'direction': -random_axis
    }

def random_vector(time: torch.Tensor, num_samples: int):

    return torch.einsum(
        'ij..., ij... -> ij...',
        angle_distribution.at(time = time).sample(num_samples = num_samples),
        axis_distribution.at(time = time).sample(num_samples = num_samples)
    ).flatten(0, 1)
