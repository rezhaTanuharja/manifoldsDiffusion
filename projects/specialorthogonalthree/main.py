from diffusionmodels import manifolds
from diffusionmodels.eulerian import stochasticprocesses as sp

import torch
import numpy as np

import matplotlib.pyplot as plt


manifold = manifolds.SpecialOrthogonal3()

angles = sp.univariate.InverseTransform(

    distribution_function = sp.univariate.functions.periodic.HeatKernel(
        num_waves = 9000,
        mean_squared_displacement = lambda t: 0.15 * t ** 4
    ),

    inversion_method = sp.univariate.inversion.Bisection(
        num_iterations = 10
    )

)

axis = sp.multivariate.UniformSphere(dimension = 3)


num_samples = 2000
time = torch.tensor([0.0, 0.5, 1.5, 2.0])

initial_value_1 = torch.tensor([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
])

initial_value_2 = torch.tensor([
    [1.0,  0.0, 0.0],
    [0.0,  1.0, 0.0],
    [0.0,  0.0, 1.0],
])

axis_angles = torch.einsum(
    'ij..., ij... -> ij...',
    angles.at(time = time).sample(num_samples = num_samples),
    axis.at(time = time).sample(num_samples = num_samples)
)

axis_angles_1 = manifold.log(
    torch.eye(3),
    manifold.exp(initial_value_1, axis_angles)
)

axis_angles_2 = manifold.log(
    torch.eye(3),
    manifold.exp(initial_value_2, axis_angles)
)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 1 * np.pi, 100)
x = np.pi * np.outer(np.cos(u), np.sin(v))
y = np.pi * np.outer(np.sin(u), np.sin(v))
z = np.pi * np.outer(np.ones(np.size(u)), np.cos(v))

for i in range(time.numel()):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    coordinates_1 = axis_angles_1[i]

    x_values_1 = coordinates_1[:, 0]
    y_values_1 = coordinates_1[:, 1]
    z_values_1 = coordinates_1[:, 2]

    ax.scatter(x_values_1, y_values_1, z_values_1, alpha = 0.10, marker = 'x')

    coordinates_2 = axis_angles_2[i]

    x_values_2 = coordinates_2[:, 0]
    y_values_2 = coordinates_2[:, 1]
    z_values_2 = coordinates_2[:, 2]

    ax.scatter(x_values_2, y_values_2, z_values_2, alpha = 0.10, marker = 'x')

    ax.set_xlim([-3.15, 3.15])
    ax.set_ylim([-3.15, 3.15])
    ax.set_zlim([-3.15, 3.15])

    ax.plot_surface(x, y, z, color = 'black', alpha = 0.05)
    ax.set_aspect('equal')

plt.show()

