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


# -- define various training and model parameters, e.g., number of data

file_name = './extractedData/ACCAD/Male1General_c3d/General A2 - Sway_poses.npz'
num_samples = 100
dim = 3

num_time_samples = 128
time_increment = 0.002

batch_size = 64
num_epochs = 50
num_super_epochs = 150

hidden_size = 512

learning_rate = 0.0002
max_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- define problem geometry and governing equation

manifold = SpecialOrthogonal3()
sde = ExplodingRotationVariance(manifold) # TODO change name to ExplodingVariance
time_integrator = EulerMaruyama()
sampler = SimpleSampler(time_integrator, SimpleRecorder())


# -- load AMASS datasets and preprocess into rotational matrices

axis_angle_data = np.load(file_name)['poses']
axis_angle_data = torch.tensor(axis_angle_data[0:num_samples], device = device).float()

num_joints = axis_angle_data.numel() // (num_samples * dim)

axis_angle_data = axis_angle_data.view(num_samples, num_joints, dim)
identity = torch.eye(dim, device = device).reshape(1, 1, dim, dim).expand(1, num_joints, dim, dim)

rotations = manifold.exp(identity, axis_angle_data)

# -- build an MLP model

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

input_size = rotations.numel() // num_samples
output_size = axis_angle_data.numel() // num_samples

model = MLP(
    input_size = input_size,
    hidden_size = hidden_size,
    output_size = output_size
)

model.load_state_dict(torch.load('model_norm.pth'))

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# -- forward noising process

for j in range(num_super_epochs):

    noised_rotations = sampler.get_samples(
        sde = sde,
        initial_condition = rotations,
        num_samples = num_time_samples,
        dt = time_increment
    )


# -- prepare dataset for training

    inverse_time_stamps = 1.0 / (time_increment * torch.arange(1, num_time_samples + 1, device = device)).repeat_interleave(num_samples)
    inverse_time_stamps = inverse_time_stamps.view(num_time_samples * num_samples, 1)

    directions = manifold.log(
        noised_rotations,
        rotations.expand(num_time_samples, num_samples, num_joints, dim, dim)
    )

    directions = directions.view(num_time_samples * num_samples, num_joints * dim)
    directions = directions * torch.sqrt(inverse_time_stamps)

    # directions_norm = torch.norm(directions, p = 2, dim = 1)
    #
    # print(directions_norm)

    noised_rotations = noised_rotations.view(
        num_time_samples * num_samples, num_joints * dim * dim
    )

    # noised_rotations = torch.cat((noised_rotations, 1.0 / inverse_time_stamps), dim = -1).contiguous()

    train_dataset = TensorDataset(noised_rotations, inverse_time_stamps, directions)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
#
#
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for i, (inputs, times, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(inputs, times)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)

            optimizer.step()

            running_loss += loss.item()

    print(f'Epoch [{j + 1}/{num_super_epochs}], Loss: {(running_loss)/len(train_loader):.4f}')


torch.save(model.state_dict(), 'model_norm.pth')
