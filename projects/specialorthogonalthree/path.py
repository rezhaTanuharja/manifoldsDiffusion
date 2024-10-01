from diffusionmodels.eulerian import stochasticprocesses as sp

import torch

import matplotlib.pyplot as plt

num_iter = 100
num_samples = 100
final_time = 2.5

cumulative_distribution_function = sp.univariate.functions.periodic.HeatKernel(
    num_waves = 9000,
    mean_squared_displacement = lambda t: 0.15 * t ** 4
)

initial_values = torch.pi * torch.rand(size = (num_samples, 1))

processes = torch.zeros(size = (num_iter + 1, *initial_values.shape))
processes[0] = initial_values

time = [final_time - i / num_iter * final_time for i in range(num_iter + 1)]
delta_time = final_time / num_iter

for i in range(num_iter):


    cumulative_distribution_function = cumulative_distribution_function.at(time = torch.tensor([time[i]]))

    points = processes[i]
    angles = points - torch.sin(points)

    denominator = (1.0 - torch.cos(points)) * cumulative_distribution_function.gradient(angles)

    velocity = 0.6 * time[i] ** 3 * (
        cumulative_distribution_function.hessian(angles) / denominator
    )

    processes[i + 1] = processes[i] + delta_time * velocity

processes = processes.view(num_iter + 1, num_samples)
plt.plot([final_time - t for t in time], processes, color = 'black', alpha = 0.2)
plt.show()
