import diffusionmodels as dm
import matplotlib.pyplot as plt
import torch


# def noiser(dataset: torch.Tensor) -> torch.Tensor:
#
#     axis_distribution = dm.distributions.UniformSphere(dimension = 3)
#
#     angle_distribution = dm.distributions.InverseTransform(
#
#         distribution_function = dm.distributions.functions.periodic.HeatKernel(
#             num_waves = 10,
#             mean_squared_displacement = lambda t: t ** 2
#         ),
#
#         inversion_method = dm.distributions.inversion.Bisection(num_iterations = 5)
#     )
#
#     time = torch.rand(size = (1,))
#
#     axis = axis_distribution.sample(num_samples = num_samples)
#
#     return dataset
#
#
# def main():
#
#     if not torch.cuda.is_available():
#         print('Requires CUDA, aborting...')
#         return
#     device = torch.device('cuda')
#
#     manifold = dm.manifolds.SpecialOrthogonal3()
#
#     for obj in [
#         manifold,
#     ]:
#         obj.to(device)
#
#     data_pipeline = dm.dataprocessing.Pipeline(
#         transforms = [
#
#             # move to device
#             lambda dataset: dataset.to(device, non_blocking = True),
#
#             # separate joints
#             lambda dataset: dataset.unflatten(-1, (-1, *manifold.tangent_dimension())),
#
#             # convert into rotational matrices
#             lambda dataset: manifold.exp(
#                 torch.eye(3, device = device).view(*manifold.dimension()),
#                 dataset
#             ),
#
#             # NOTE: the number of samples here is still a placeholder
#
#             # duplicate to process with multiple sample paths simultaneously
#             lambda dataset: dataset.unsqueeze(0).expand(
#                 50, *dataset.shape
#             ).flatten(0,1),
#
#             # TODO: create a noiser function
#             # it should return a dictionary with 'time', 'points', and 'labels' as keys
#
#         ]
#     )
#
#
#
# if __name__ == '__main__':
#     main()



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

num_samples = 1000

# axis_distribution = dm.distributions.UniformSphere(dimension = 3)
angle_distribution = dm.distributions.univariate.InverseTransform(

    distribution_function = dm.distributions.univariate.functions.periodic.HeatKernel(
        num_waves = 10,
        mean_squared_displacement = lambda t: t
    ),

    inversion_method = dm.distributions.univariate.inversion.Bisection(num_iterations = 5)

)

# angles = torch.pi * torch.arange(start = 0.0, end = 1.0, step = 0.1)
# values = angle_distribution.density_function()(angles)

values = angle_distribution.at(time = torch.tensor([10.0, 1.0, 0.2])).sample(num_samples)

for i in range(values.shape[0]):

    plt.hist(values[i])

plt.show()

# plt.hist(angles)


# axis_angle_distribution = dm.distributions.Separable(
#     distributions = (axis_distribution, angle_distribution)
# )


# random_axis = axis_distribution.sample(num_samples = num_samples)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# random_angle = 1.0 + torch.abs(angle_distribution.at(time = 1.2).sample(num_samples = num_samples))

# axis_angle = torch.einsum(
#     'i..., i... -> i...',
#     random_axis,
#     random_angle
# )

# axis_angle = axis_angle_distribution.sample(num_samples = num_samples)
#
# x = axis_angle[:, 0]
# y = axis_angle[:, 1]
# z = axis_angle[:, 2]
#
# ax.scatter(x, y, z, color = 'black')

# times = [0.25 * i for i in range(10)]
#
# for t in times:
#     random_angle = 1.0 + torch.abs(angle_distribution.at(time = t).sample(num_samples = num_samples))
#
#     axis_angle = torch.einsum(
#         'i..., i... -> i...',
#         random_axis,
#         random_angle
#     )
#
#     x = axis_angle[:, 0]
#     y = axis_angle[:, 1]
#     z = axis_angle[:, 2]
#
#     ax.scatter(x, y, z, color = 'black', alpha = (3.0 - t) / 6.0)

# ax.set_aspect('equal')

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
# plt.show()
