import numpy as np
import torch

from diffusionmodels.manifolds import SpecialOrthogonal3
from diffusionmodels.differentialequations import ExplodingVariance
from diffusionmodels.timeintegrators import EulerMaruyama
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler

import torch.nn as nn


# -- describe parameters
num_samples = 1
num_time_samples = 1
dim = 3

time_increment = 0.002

hidden_size = 512


# -- define problem geometry and governing equations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifold = SpecialOrthogonal3(device = device)
sde = ExplodingVariance(manifold) # TODO change name to ExplodingVariance
time_integrator = EulerMaruyama()
sampler = SimpleSampler(time_integrator, SimpleRecorder())


# -- load noised data from a previously generated npz file
file_name = './downloads/processed_A2.npz'

axis_angle_data = np.load(file_name)['poses']
axis_angle_data = axis_angle_data[-num_time_samples:]

num_joints = axis_angle_data.size // (num_time_samples * num_samples * dim)


# -- convert into torch tensor and move to an appropriate device

noised_data = torch.tensor(axis_angle_data, device = device)
noised_data = noised_data.view(num_time_samples * num_samples, num_joints, dim)

identity = torch.eye(dim, device = device).reshape(1, 1, dim, dim).expand(num_time_samples * num_samples, num_joints, dim, dim)

noised_rotations = manifold.exp(identity, noised_data)
noised_rotations = noised_rotations.view(
    num_time_samples * num_samples, num_joints * dim * dim
)

noised = noised_rotations.view(num_time_samples, num_samples, num_joints, dim, dim)

times = torch.tensor(128 * time_increment).to(device)
times = times.view(num_time_samples * num_samples, 1)


# -- define the model and load from pretrained model
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fct = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.01)

        self.fct.bias.data.fill_(0)
        self.fct.bias.requires_grad = False

    def forward(self, x, t):
        x = self.leaky_relu(self.fc1(x) + self.fct(t))
        x = self.leaky_relu(self.fc2(x))
        # x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_size = noised_rotations.numel() // num_samples
output_size = noised_data.numel() // num_samples

model = MLP(
    input_size = input_size,
    hidden_size = hidden_size,
    output_size = output_size
)

model.load_state_dict(torch.load('model_norm.pth'))
model.to(device)

model.eval()


# -- test run with one parameters

for _ in range(128):

    # print(times[0])
    output = model(noised_rotations, 1.0 / times)
    # output = time_increment * output
    # output = time_increment / torch.sqrt(times) * output
    # output = time_increment * torch.sqrt(times) * output
    output = time_increment * (times ** 0.4) * output
    output = output.view(num_time_samples, num_samples, num_joints, dim)

    noised = manifold.exp(noised, output)
    times = times - time_increment


noised = manifold.log(identity, noised)
noised = noised.view(num_time_samples * num_samples, num_joints * dim)

numpy_array = noised.detach().cpu().numpy()
np.savez('downloads/postprocessed_A2.npz', poses=numpy_array)
