import diffusionmodels as dm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


manifold = dm.manifolds.SpecialOrthogonal3()

magnitude_distribution = dm.processes.eulerian.distributions.univariate.InverseTransform(
    distribution_function = dm.processes.eulerian.distributions.univariate.functions.periodic.HeatKernel(
        num_waves = 10,
        mean_squared_displacement = lambda t: t
    ),
    inversion_method = dm.processes.eulerian.distributions.univariate.inversion.Bisection(num_iterations = 8)
)

distribution_function = dm.processes.eulerian.distributions.univariate.functions.Normalizer()
inversion_method = dm.processes.eulerian.distributions.univariate.inversion.Bisection(num_iterations = 8)

direction_distribution = dm.processes.eulerian.distributions.multivariate.UniformSphere(dimension = 3)

# random_vector = dm.processes.eulerian.forward.RandomVector(
#     magnitude_distribution = magnitude_distribution,
#     direction_distribution = direction_distribution
# )

# random_flow = dm.processes.eulerian.forward.RandomFlow(
#     manifold = manifold,
#     vector_distribution = random_vector
# )

time = torch.tensor([0.0, 0.3, 50.0])

# vector = random_vector.at(
#     time = time
# ).sample(num_samples = 5)


# points = random_flow.at(time = time).sample(
#     initial_value = torch.tensor([
#         [1.0,  0.0, 0.0],
#         [0.0,  1.0, 0.0],
#         [0.0,  0.0, 1.0]
#     ]),
#     num_samples = 2000
# )

magnitudes = torch.abs(magnitude_distribution.at(time = time).sample(num_samples=2000))

# angles = inversion_method.solve(magnitudes, distribution_function.cumulative, distribution_function.support())

angles = torch.zeros(magnitudes.shape)
for _ in range(12):
    angles = magnitudes + torch.sin(angles)



directions = direction_distribution.at(time = time).sample(num_samples=2000)

axangle = torch.einsum('ij...,ij...->ij...', angles, directions)

points = manifold.exp(torch.tensor([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
]), axangle)

axangle = manifold.log(torch.eye(3), points)

# plt.hist(angles[2])
# plt.hist(angles[1])
# plt.hist(angles[0])

# axangle = manifold.log(torch.eye(3), points)
#
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
#
# # axs = [
# #     fig.add_subplot(131, projection = '3d'),
# #     fig.add_subplot(132, projection = '3d'),
# #     fig.add_subplot(133, projection = '3d'),
# # ]

# ax2 = fig.add_subplot(111, projection = '3d')
# x = axangle[0,:,0]
# y = axangle[0,:,1]
# z = axangle[0,:,2]
# ax2.scatter(x, y, z, alpha = 0.05)

#
# for i in range(time.numel()):
#
#     axang = axangle[i]
#
#     x = axang[:, 0]
#     y = axang[:, 1]
#     z = axang[:, 2]
#
#     ax.scatter(x, y, z, alpha = 0.05)
#     ax.set_aspect('equal')
#     ax.set_xlim([-3.15, 3.15])
#     ax.set_ylim([-3.15, 3.15])
#     ax.set_zlim([-3.15, 3.15])
    # axs[i].axes.set_ylim(left = -3.15, right = 3.15)
    # axs[i].axes.set_zlim(left = -3.15, right = 3.15)
    # axs[i].axes.set_ylim3d([-3.15, 3.15])
    # axs[i].axes.set_zlim([-3.15, 3.15])

plt.show()
