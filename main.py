import diffusionmodels as dm
import matplotlib.pyplot as plt
import torch


device = torch.device('cpu')

distribution = dm.distributions.InverseTransform(
    # cumulative_distribution_function = dm.distributions.functions.Heaviside(),
    distribution_function = dm.distributions.functions.periodic.HeatKernel(
        num_waves = 10,
        mean_squared_displacement = lambda t: t
    ),
    inversion_method = dm.distributions.inversion.Bisection(num_iterations = 10)
)
# cumulative_distribution_function = dm.distributions.functions.periodic.HeatKernel(
#     num_waves = 10,
#     mean_squared_displacement = lambda t: 0.375 * t ** 2
# )
#
# angles = torch.pi * torch.arange(start = -1.0, end = 1.0, step = 0.01)
# angles.to(device)

# times = [0.3 * (1.0 + i) for i in range(10)]

# for t in times:
#     values = cumulative_distribution_function.at(time = t).density(angles)
#     plt.plot(angles.cpu(), values.cpu())

random_samples = distribution.at(time = 1.0).sample(num_samples = 2000)

plt.hist(random_samples.cpu())
plt.show()
# distribution.to(device)

# points = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
# points = points.unflatten(0, (-1, 1))

# values = torch.rand(size = (1000, 1))
#
# random_samples = inverter.solve(
#     values = values,
#     function = cdf.at(time = 0)
# )
#
# random_samples = random_samples.view(1000)

# plt.xlim([-4., 4.])

