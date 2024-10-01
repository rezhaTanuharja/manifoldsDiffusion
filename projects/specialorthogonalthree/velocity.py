from diffusionmodels import manifolds
from diffusionmodels.eulerian import stochasticprocesses as sp

import torch

import matplotlib.pyplot as plt

manifold = manifolds.SpecialOrthogonal3()


cumulative_distribution_function = sp.univariate.functions.periodic.HeatKernel(
    num_waves = 9000,
    mean_squared_displacement = lambda t: 0.15 * t ** 4
)

angles_distribution = sp.univariate.InverseTransform(

    cumulative_distribution_function = cumulative_distribution_function,

    inversion_method = sp.univariate.inversion.Bisection(
        num_iterations = 10,
        pretransform = lambda points: points - torch.sin(points)
    )

)

time = torch.arange(start = 0.2, end = 1.5, step = 0.10)

angles = angles_distribution.at(time = time).sample(num_samples = 1000)

time_tensor = time.view(time.numel(), 1)

alpha = angles - torch.sin(angles)

cdf_values = cumulative_distribution_function.at(time = time)(alpha)
pdf_values = cumulative_distribution_function.at(time = time).gradient(alpha)
hessian_values = cumulative_distribution_function.at(time = time).hessian(alpha)
denominator_values = (1.0 - torch.cos(angles)) * cumulative_distribution_function.at(time = time).gradient(alpha)


velocities = 0.6 * time_tensor ** 4.0 * (
    -hessian_values
    /
    denominator_values
)

for i in range(time.numel()):
    # plt.scatter(angles[i], pdf_values[i], marker = 'x', alpha = 0.2)
    # plt.plot(angles[0], density_values[i])

    speed = velocities[i, velocities[i] > 1e-8]
    plt.scatter(angles[i, velocities[i] > 1e-8], speed, marker = 'x', s = 3, alpha = 0.2)
plt.show()
