from torch import random
import diffusionmodels as dm
import torch
# import matplotlib
#
# matplotlib.use('Gtk3Cairo')
import matplotlib.pyplot as plt


cdf = dm.distributions.functions.Linear(1.0, 3.0)
inverter = dm.distributions.inversion.Bisection(num_iterations = 50)

# points = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
# points = points.unflatten(0, (-1, 1))

values = torch.rand(size = (1000, 1))

random_samples = inverter.solve(
    values = values,
    function = cdf.at(time = 0)
)

random_samples = random_samples.view(1000)

plt.hist(random_samples)
plt.show()

