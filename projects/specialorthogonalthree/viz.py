from diffusionmodels import manifolds
from diffusionmodels.eulerian import stochasticprocesses as sp

import torch
import numpy as np

import matplotlib.pyplot as plt


# -- set the number of samples and sampling time

num_samples = 2000
time = torch.tensor([0.0, 0.5, 1.5, 2.5])


# -- set distribution for angles and axis

angles_distribution = sp.univariate.InverseTransform(

    cumulative_distribution_function = sp.univariate.functions.periodic.HeatKernel(
        num_waves = 1000,
        mean_squared_displacement = lambda t: 0.15 * t ** 4
    ),

    inversion_method = sp.univariate.inversion.Bisection(
        num_iterations = 12,
        pretransform = lambda points: points - torch.sin(points)
    )

)

axis_distribution = sp.multivariate.UniformSphere(dimension = 3)

# -- sample random vector from the predefined distributions

random_vector = torch.einsum(
    'ij..., ij... -> ij...',
    angles_distribution.at(time = time).sample(num_samples = num_samples),
    axis_distribution.at(time = time).sample(num_samples = num_samples)
)

# -- define two initial values for a synthetic distribution

phi = torch.tensor(torch.pi - 0.25)

initial_value_1 = torch.tensor([
    [torch.cos(phi), -torch.sin(phi), 0.0],
    [torch.sin(phi),  torch.cos(phi), 0.0],
    [           0.0,             0.0, 1.0],
])

initial_value_2 = torch.eye(3)

# -- transform into an axis-angle representation for visualization

manifold = manifolds.SpecialOrthogonal3()

points_1 = manifold.log(
    torch.eye(3),
    manifold.exp(initial_value_1, random_vector)
)

points_2 = manifold.log(
    torch.eye(3),
    manifold.exp(initial_value_2, random_vector)
)

# -- this part is to define the bounding sphere

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 1 * np.pi, 100)
x = np.pi * np.outer(np.cos(u), np.sin(v))
y = np.pi * np.outer(np.sin(u), np.sin(v))
z = np.pi * np.outer(np.ones(np.size(u)), np.cos(v))

for i in range(time.numel()):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    coordinates_1 = points_1[i]

    x_values_1 = coordinates_1[:, 0]
    y_values_1 = coordinates_1[:, 1]
    z_values_1 = coordinates_1[:, 2]

    ax.scatter(x_values_1, y_values_1, z_values_1, alpha = 0.10, marker = 'x', s = 2)

    coordinates_2 = points_2[i]

    x_values_2 = coordinates_2[:, 0]
    y_values_2 = coordinates_2[:, 1]
    z_values_2 = coordinates_2[:, 2]

    ax.scatter(x_values_2, y_values_2, z_values_2, alpha = 0.10, marker = 'x', s = 2)

    ax.set_xlim([-3.15, 3.15])
    ax.set_ylim([-3.15, 3.15])
    ax.set_zlim([-3.15, 3.15])

    ax.plot_surface(x, y, z, color = 'black', alpha = 0.05)
    ax.set_aspect('equal')

plt.show()
