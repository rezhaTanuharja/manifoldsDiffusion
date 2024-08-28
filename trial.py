import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusionmodels.manifolds import SpecialOrthogonal3
from diffusionmodels.differentialequations import ExplodingRotationVariance
from diffusionmodels.timeintegrators import EulerMaruyama
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler

num_samples = 1
num_time_samples = 1
dim = 3

time_increment = 0.0002
hidden_size = 512

file_name = './downloads/processed_A2.npz'
data = np.load(file_name)['poses']
data = data[-num_samples:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noised_data = torch.tensor(data, device = device)
noised_data = noised_data.view(1, 52, 3)


# -- define problem geometry and governing equation

manifold = SpecialOrthogonal3()
sde = ExplodingRotationVariance(manifold) # TODO change name to ExplodingVariance
time_integrator = EulerMaruyama()
sampler = SimpleSampler(time_integrator, SimpleRecorder())


# -- load AMASS datasets and preprocess into rotational matrices

num_joints = noised_data.numel() // (num_samples * dim)
identity = torch.eye(dim, device = device).reshape(1, 1, dim, dim).expand(1, num_joints, dim, dim)

noised_rotations = manifold.exp(identity, noised_data)
noised_rotations = noised_rotations.view(
    num_time_samples * num_samples, num_joints * dim * dim
)

noised_rotations = torch.cat((noised_rotations, torch.ones(num_time_samples * num_samples, 1, device = device)), dim = 1)

# -- build an MLP model

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
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

output = model(noised_rotations)
print(output)
