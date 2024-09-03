import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import diffusionmodels as dm

# TODO: fix this please
# NOTE: this is not a joke

num_samples = 1000
num_time_samples = 128
batch_size = 250
time_batch = 512
num_epochs = 2500
max_norm = 3.0
repeat_noise = 10
learning_rate = 0.00005
file_name = './extractedData/ACCAD/Male1General_c3d/General A2 - Sway_poses.npz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_increment = 0.002

special_orthogonal_3 = dm.manifolds.SpecialOrthogonal3(device = device)

exploding_variance = dm.differentialequations.ExplodingVariance(
    manifold = special_orthogonal_3,
    variance_scale = time_increment ** 0.5
)

solver = dm.solvers.SimpleSolver(
    time_integrator = dm.timeintegrators.EulerMaruyama(),
    data_recorder = dm.recorders.SimpleRecorder(device = device),
    num_points = num_time_samples,
    grid_size = time_increment
)

direction = dm.scorefunctions.Direction(manifold = special_orthogonal_3)

amass_dataset = np.load(file_name)['poses']
amass_dataset = torch.tensor(amass_dataset[:num_samples]).float()

tensor_dataset = TensorDataset(amass_dataset)
data_loader = DataLoader(amass_dataset, batch_size = batch_size, shuffle = True)



data_pipeline = dm.pipeline.Pipeline(
    transforms = [

        # -- move dataset to the available device
        lambda dataset: dataset.to(device),
        lambda dataset: dataset.unflatten(-1, (-1, *special_orthogonal_3.tangent_dimension())),

        lambda dataset: special_orthogonal_3.exp(
            torch.eye(3, device = device).view(*special_orthogonal_3.dimension()),
            dataset
        ),

        lambda dataset: dataset.unsqueeze(0).expand(repeat_noise, *dataset.shape).flatten(0,1),

        lambda dataset: {
            **(solver.solve(dataset, exploding_variance)),
            'original': dataset
        },

        lambda dataset: {

            'time': dataset['time'].reshape(*dataset['time'].shape, 1).expand(-1, repeat_noise * batch_size),

            'points': dataset['data'],

            'labels': direction.get_direction(
                origin = dataset['data'],
                destination = dataset['original'],
                scale = 1.0 / dataset['time']
            )
        },

        lambda dataset: {
            'time': dataset['time'].flatten(0, 1).unsqueeze(-1),
            'points': dataset['points'].flatten(0, 1).flatten(1),
            'labels': dataset['labels'].flatten(0, 1).flatten(1)
            # 'time': dataset['time'].flatten(0, 1)
            # key: value.flatten(0, 1).flatten(1) for key, value in dataset.items()
        },

    ]
)

class MLP(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(468, hidden_size)
        self.fct = nn.Linear(1, hidden_size, bias = False)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fcf = nn.Linear(hidden_size, output_size)

        self.leaky_ReLU = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x, t):
        x = self.leaky_ReLU(self.fc1(x) + self.fct(100.0  * t))
        x = self.leaky_ReLU(self.fc2(x))

        x = self.fcf(x)
        return x

model = MLP(hidden_size=512, output_size=156)
model.load_state_dict(torch.load('model_norm.pth'))
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for k in range(num_epochs):

    model.train()
    running_loss = 0.0

    for i, dataset in enumerate(data_loader):

        # for _ in range(repeat_noise):
        data = data_pipeline(dataset)
        # print(data['time'].shape)
        # print(data['points'].shape)
        # print(data['labels'].shape)

        train_data = TensorDataset(data['time'], data['points'], data['labels'])
        train_load = DataLoader(train_data, batch_size = time_batch, shuffle = True)


        for j, (time, point, label) in enumerate(train_load):

            optimizer.zero_grad()

            outputs = model(point, time)
            # print(outputs)
            # print(label)

            loss = criterion(outputs, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)
            optimizer.step()

            running_loss += loss.item()

    print(f'Epoch [{k + 1}/{num_epochs}], Loss: {(running_loss / (len(data_loader) * len(train_load))):.4f}')

    if k % 500 == 0:
        torch.save(model.state_dict(), 'model_norm.pth')
