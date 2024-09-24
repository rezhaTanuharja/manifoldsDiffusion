from torch import random
import diffusionmodels as dm
import torch
# import matplotlib
#
# matplotlib.use('Gtk3Cairo')
import matplotlib.pyplot as plt


device = torch.device('cpu')

# distribution = dm.distributions.MultivariateGaussian(dimension=1)
distribution = dm.distributions.InverseTransform(
    cumulative_distribution_function = dm.distributions.functions.StepFunction(100),
    inversion_method = dm.distributions.inversion.Bisection(num_iterations = 10)
)
# cumulative_distribution_function = dm.distributions.functions.StepFunction(10)
# angles = torch.pi * torch.arange(start = 0, end = 1.0, step = 0.01)
# values = cumulative_distribution_function(angles)

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
random_samples = distribution.sample(num_samples = 2000)

plt.hist(random_samples.cpu())
# plt.plot(angles.cpu(), values.cpu())
plt.show()

