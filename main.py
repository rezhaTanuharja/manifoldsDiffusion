import diffusionmodels as dm
import matplotlib.pyplot as plt
import torch


device = torch.device('cpu')

distribution = dm.distributions.InverseTransform(

    distribution_function = dm.distributions.functions.periodic.HeatKernel(
        num_waves = 10,
        mean_squared_displacement = lambda t: 0.375 * t
    ),
    
    inversion_method = dm.distributions.inversion.Bisection(num_iterations = 10)

)

angles = torch.pi * torch.arange(start = -1.0, end = 1.0, step = 0.01)
angles = angles.to(device)

times = [0.125 * (1.0 + i) for i in range(20)]

for t in times:
    values = distribution.at(time = t).density_function()(angles)
    plt.plot(angles.cpu(), values.cpu())

plt.show()
