import diffusionmodels as dm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from diffusionmodels import manifolds

manifold = dm.manifolds.SpecialOrthogonal3()


uniform_samples = torch.rand(size = (1, 1000))

cdf = dm.eulerian.distributions.univariate.functions.periodic.HeatKernel(
    num_waves = 9000,
    mean_squared_displacement = lambda t: 0.15 * t ** 4
)

inverter_1 = dm.eulerian.distributions.univariate.inversion.Bisection(
    num_iterations = 10
)

inverter_2 = dm.eulerian.distributions.univariate.inversion.Newton(
    max_iter = 8,
    tolerance = 1e-6
)

inverter_3 = dm.eulerian.distributions.univariate.inversion.Secant(
    max_iter = 8,
    tolerance = 1e-6
)

# time = torch.tensor([0.0, 0.15, 0.3, 0.45, 0.6])
time = torch.arange(start = 0.0, end = 2.75, step = 0.25)
# time = torch.tensor([10.02])
cdf = cdf.at(time = time)

angles_1 = inverter_1.solve(
    values = uniform_samples,
    function = cdf,
    search_range = cdf.support()
)

# density_1 = cdf.gradient(angles_1)
direction_distribution = dm.eulerian.distributions.multivariate.UniformSphere(dimension = 3)
directions = direction_distribution.at(time = time).sample(1000)

axangle = torch.einsum('ij...,ij...->ij...', angles_1, directions)

rot_matrix = manifold.exp(torch.tensor([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
]), axangle)

axangle = manifold.log(torch.eye(3), rot_matrix)

# angles_2 = inverter_2.solve(
#     values = uniform_samples,
#     function = cdf,
#     search_range = cdf.support()
# )
#
# density_2 = cdf.gradient(angles_2)

# angles_3 = inverter_3.solve(
#     values = uniform_samples,
#     function = cdf,
#     search_range = cdf.support()
# )
#
# density_3 = cdf.gradient(angles_3)

# values = torch.pi * uniform_samples
# # asymptote = (values - torch.sin(values)) / torch.pi
# asymptote = (1.0 - torch.cos(values)) / torch.pi
#
# for i in range(time.numel()):
#
#     plt.scatter(angles_1[i], density_1[i], alpha = 0.02, color = 'black', marker = '+')
#     # plt.scatter(angles_2[i], density_2[i], alpha = 0.2, color = 'red', marker = '+')
#     # plt.scatter(angles_3[i], density_3[i], alpha = 0.2, color = 'blue', marker = '+')
#     # plt.scatter(angles_1[i], uniform_samples, alpha = 0.2, color = 'black', marker = '+')
#     # plt.scatter(angles_2[i], uniform_samples, alpha = 0.2, color = 'red', marker = '+')
#     # plt.scatter(angles_3[i], uniform_samples, alpha = 0.2, color = 'blue', marker = '+')
#
# # plt.scatter(values, asymptote, alpha = 0.1, color = 'green', marker = 'x')
# plt.scatter(values, asymptote, alpha = 0.1, color = 'green', marker = 'x')
# plt.show()

# manifold = dm.manifolds.SpecialOrthogonal3()
#
# magnitude_distribution = dm.processes.eulerian.distributions.univariate.InverseTransform(
#     distribution_function = dm.processes.eulerian.distributions.univariate.functions.periodic.HeatKernel(
#         num_waves = 10,
#         mean_squared_displacement = lambda t: t
#     ),
#     inversion_method = dm.processes.eulerian.distributions.univariate.inversion.Bisection(num_iterations = 8)
# )
#
# distribution_function = dm.processes.eulerian.distributions.univariate.functions.Normalizer()
# inversion_method = dm.processes.eulerian.distributions.univariate.inversion.Bisection(num_iterations = 8)
#
# direction_distribution = dm.processes.eulerian.distributions.multivariate.UniformSphere(dimension = 3)
#
# # random_vector = dm.processes.eulerian.forward.RandomVector(
# #     magnitude_distribution = magnitude_distribution,
# #     direction_distribution = direction_distribution
# # )
#
# # random_flow = dm.processes.eulerian.forward.RandomFlow(
# #     manifold = manifold,
# #     vector_distribution = random_vector
# # )
#
# time = torch.tensor([0.0, 0.3, 50.0])
#
# # vector = random_vector.at(
# #     time = time
# # ).sample(num_samples = 5)
#
#
# # points = random_flow.at(time = time).sample(
# #     initial_value = torch.tensor([
# #         [1.0,  0.0, 0.0],
# #         [0.0,  1.0, 0.0],
# #         [0.0,  0.0, 1.0]
# #     ]),
# #     num_samples = 2000
# # )
#
# magnitudes = torch.abs(magnitude_distribution.at(time = time).sample(num_samples=2000))
#
# # angles = inversion_method.solve(magnitudes, distribution_function.cumulative, distribution_function.support())
#
# angles = torch.zeros(magnitudes.shape)
# for _ in range(12):
#     angles = magnitudes + torch.sin(angles)
#
#
#
# directions = direction_distribution.at(time = time).sample(num_samples=2000)
#
# axangle = torch.einsum('ij...,ij...->ij...', angles, directions)
#
# points = manifold.exp(torch.tensor([
#     [0.0, -1.0, 0.0],
#     [1.0,  0.0, 0.0],
#     [0.0,  0.0, 1.0]
# ]), axangle)
#
# axangle = manifold.log(torch.eye(3), points)
#
# # plt.hist(angles[2])
# # plt.hist(angles[1])
# # plt.hist(angles[0])
#
# # axangle = manifold.log(torch.eye(3), points)
# #
for i in range(time.numel()):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    x = axangle[i,:,0]
    y = axangle[i,:,1]
    z = axangle[i,:,2]
    ax.scatter(x, y, z, alpha = 0.05)
    ax.set_xlim([-3.15, 3.15])
    ax.set_ylim([-3.15, 3.15])
    ax.set_zlim([-3.15, 3.15])
    ax.set_aspect('equal')
# #
# # # axs = [
# # #     fig.add_subplot(131, projection = '3d'),
# # #     fig.add_subplot(132, projection = '3d'),
# # #     fig.add_subplot(133, projection = '3d'),
# # # ]
#
# # ax2 = fig.add_subplot(111, projection = '3d')
# # x = axangle[0,:,0]
# # y = axangle[0,:,1]
# # z = axangle[0,:,2]
# # ax2.scatter(x, y, z, alpha = 0.05)
#
# #
# # for i in range(time.numel()):
# #
# #     axang = axangle[i]
# #
# #     x = axang[:, 0]
# #     y = axang[:, 1]
# #     z = axang[:, 2]
# #
# #     ax.scatter(x, y, z, alpha = 0.05)
# #     ax.set_aspect('equal')
# #     ax.set_xlim([-3.15, 3.15])
# #     ax.set_ylim([-3.15, 3.15])
# #     ax.set_zlim([-3.15, 3.15])
#     # axs[i].axes.set_ylim(left = -3.15, right = 3.15)
#     # axs[i].axes.set_zlim(left = -3.15, right = 3.15)
#     # axs[i].axes.set_ylim3d([-3.15, 3.15])
#     # axs[i].axes.set_zlim([-3.15, 3.15])
#
plt.show()
