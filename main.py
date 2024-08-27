import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusionmodels.manifolds import SpecialOrthogonal3
from diffusionmodels.differentialequations import ExplodingRotationVariance, CorrectedNegative
from diffusionmodels.timeintegrators import EulerMaruyama
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler
from diffusionmodels.scorefunctions import DirectToReference


file_name = './extractedData/ACCAD/Male1General_c3d/General A2 - Sway_poses.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_samples = 50
num_time_samples = 10


data = np.load(file_name)['poses']
data = data[0:num_samples].reshape(num_samples, 52, 3)

data = torch.tensor(data, device = device).float()
iden = torch.eye(3, device = device).reshape(1, 1, 3, 3).expand(1, 52, 3, 3)

manifold = SpecialOrthogonal3()
SDE = ExplodingRotationVariance(manifold)
time_integrator = EulerMaruyama()

sampler = SimpleSampler(time_integrator, SimpleRecorder())

rotation = manifold.exp(iden, data)
# score_function = DirectToReference(manifold, rotation, 100 * 0.003)

noisy_rotation = sampler.get_samples(
    sde = SDE,
    initial_condition = rotation,
    num_samples = num_time_samples,
    dt = 0.003
)

directions = torch.zeros(noisy_rotation.shape[:-1], device = device)

for i in range(num_time_samples):

    directions[i] = 1.0 / ((num_time_samples - i) * 0.003) * manifold.log(noisy_rotation[i], rotation)


# X = manifold.log(iden, noisy_rotation)
# X_train = torch.zeros(1000, 157, device = device)
# X_train[:, :-1] = X.view(1000, 156)
# X_train[:, -1] = 1.0 / (0.003 * torch.arange(1, 101, device = device).repeat_interleave(10))
X = noisy_rotation.view(num_samples * num_time_samples, 468)
X_train = torch.zeros(num_samples * num_time_samples, 469, device = device)
X_train[:, :-1] = X.view(num_samples * num_time_samples, 468)
X_train[:, -1] = 1.0 / (0.003 * torch.arange(1, num_samples + 1, device = device).repeat_interleave(num_time_samples))
# X_train[:, -1] = 0.003 * torch.arange(1, num_samples + 1, device = device).repeat_interleave(num_time_samples)
# X_train = X.view(1000, 156)
Y_train = directions.view(num_samples * num_time_samples, 156)
# print(Y_train[0])

# print(X_train[:, -1])
# print(X_train[0])
# print(X_train[999])

# print(X_train.shape)
# print(Y_train[0])

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        # x = self.leaky_relu(self.fc2(x))
        # x = self.leaky_relu(self.fc3(x))
        # x = self.leaky_relu(self.fc4(x))
        # x = self.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return x


model = MLP(input_size=469, hidden_size=640, output_size=156)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)


for epoch in range(30000):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(inputs)
        # print(labels)
        # some_loss = criterion(outputs, labels).mean()
        # initial_loss = criterion(outputs, labels).mean(dim=1) * inputs[:, -1]
        loss = criterion(outputs, labels)


        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
        optimizer.step()

        # print(loss.item())
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/20], Loss: {(running_loss)/len(train_loader):.4f}')
