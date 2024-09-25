import diffusionmodels as dm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch



device = torch.device('cpu')

# distribution = dm.distributions.InverseTransform(
#
#     distribution_function = dm.distributions.functions.periodic.HeatKernel(
#         num_waves = 10,
#         mean_squared_displacement = lambda t: 0.375 * t
#     ),
#
#     inversion_method = dm.distributions.inversion.Bisection(num_iterations = 10)
#
# )

distribution = dm.distributions.UniformSphere(dimension = 3)
random_variables = distribution.sample(num_samples = 1000)

x = random_variables[:, 0]
y = random_variables[:, 1]
z = random_variables[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x, y, z)
ax.set_aspect('equal')

# angles = torch.pi * torch.arange(start = -1.0, end = 1.0, step = 0.01)
# angles = angles.to(device)
#
# times = [1.0 * (1.0 + i) for i in range(10)]
#
# for t in times:
#     values = distribution.at(time = t).density_function()(angles)
    # plt.plot(angles.cpu(), values.cpu())

    # random_variables = distribution.at(time = t).sample(num_samples = 2000)
    # plt.hist(random_variables, alpha = 1.0 - 0.08 * t)

# plt.xlim([-3.15, 3.15])
plt.show()
