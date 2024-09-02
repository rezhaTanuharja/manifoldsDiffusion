import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import diffusionmodels as dm

num_samples = 1000
num_time_samples = 128
batch_size = 100
time_batch = 512
num_epochs = 2000
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
    data_recorder = dm.recorders.SimpleRecorder(),
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

        lambda dataset: {
            **(solver.solve(dataset, exploding_variance)),
            'original': dataset
        },

        lambda dataset: {

            'time': dataset['time'].reshape(*dataset['time'].shape, 1).expand(-1, batch_size),

            'points': dataset['noised'],

            'labels': direction.get_direction(
                origin = dataset['noised'],
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

        for _ in range(repeat_noise):
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

    print(f'Epoch [{k + 1}/{num_epochs}], Loss: {(running_loss / (repeat_noise * len(data_loader) * len(train_load))):.4f}')

torch.save(model.state_dict(), 'model_norm.pth')

# print(amass_dataset.device)
    # print(data['time'].shape)
    # print(data['points'].shape)
    # print(data['labels'].shape)



# # ----------------------------------------------------------------------------- #
# # -- set up the simulation ---------------------------------------------------- #
# # ----------------------------------------------------------------------------- #
#
# # -- use NVIDIA GPU whenever it is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # -- simulation config
# num_samples = 1000
# time_increment = 0.001
#
# num_joints = 52
# dim = 3
#
# hidden_size = 400
#
# file_name = './extractedData/ACCAD/Male1General_c3d/General A2 - Sway_poses.npz'
#
#
# # -- the manifold is SO(3)
# special_orthogonal_3 = dm.manifolds.SpecialOrthogonal3(device = device)
#
# # -- the stochastic differential equation is an exploding variance sde in SO3
# stochastic_de = dm.differentialequations.ExplodingVariance(
#
#     manifold = special_orthogonal_3,
#     variance_scale = time_increment ** 0.5
#
# )
#
# score_function = dm.scorefunctions.DirectToReference(manifold = special_orthogonal_3)
#
# # -- use a simple SDE solver
# solver = dm.solvers.SimpleSolver(
#     time_integrator = dm.timeintegrators.EulerMaruyama(),
#     data_recorder = dm.recorders.SimpleRecorder(),
#     num_points = num_samples,
#     grid_size = time_increment
# )
#
#
#
# class CustomTrans(dm.pipeline.Transform):
#     def __init__(self, a_function):
#         self._func = a_function
#
#     def __call__(self, x):
#         # y = [k for (k, m) in x]
#         return list(zip(*x, self._func(x)))
#
# custom_trans = CustomTrans(solver.solve)
#
# # -- define a pipeline
# generate_noisy_data = dm.pipeline.Pipeline([
#
#     # -- move datasets from cpu to device (gpu if available)
#     (lambda datasets: [dataset.to(device) for dataset in datasets]),
#
#     # -- pair each dataset with its respective sde
#     (lambda datasets: list(zip(datasets, sdes))),
#
#     # -- amass datasets are stored in the form of tangent vectors
#     # -- needs to convert it into points first
#     dm.utilities.extract_points_from_amass,
#
#     # -- forward noising process
#     # solver.solve,
#     custom_trans
#
#     # -- reshape tensor so it is fit for training
#     # (lambda datasets: [dataset.flatten(0,1).flatten(1) for dataset in datasets])
# ])
#
# # -- define a NN model
# # class MLP(nn.Module):
# #
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(MLP, self).__init__()
# #
# #         self.fc1 = nn.Linear(input_size, hidden_size)
# #         self.fct = nn.Linear(1, hidden_size)
# #
# #         self.fcf = nn.Linear(hidden_size, output_size)
# #
# #         self.leaky_ReLU = nn.LeakyReLU(negative_slope=0.01)
# #
# #     def forward(self, x, t):
# #         x = self.leaky_ReLU(self.fc1(x) + self.fct(t))
# #
# #         x = self.fcf(x)
# #         return x
# #
# # model = MLP(
# #     input_size = num_joints * dim * dim,
# #     hidden_size = hidden_size,
# #     output_size = num_joints * dim
# # )
# #
# # model.to(device)
#
# data = np.load(file_name)['poses']
# data = [torch.Tensor(data), ]
#
# # data = [
# #     torch.randn(10, 156),
# #     torch.randn(10, 156),
# # ]
#
# sdes = [
#     stochastic_de,
#     # stochastic_de,
# ]
#
#
# dataset = TensorDataset(*data)
# dloader = DataLoader(dataset, batch_size = 2, shuffle = True)
#
#
# for i, data in enumerate(dloader):
#
#     noised_rotations = generate_noisy_data(data)
#
#     print(type(noised_rotations[0][0]))
#     # print(noised_rotations[0][1].shape) 
#
#     # noised_dataset = TensorDataset(*noised_rotations)
#     # noised_dloader = DataLoader(noised_dataset, batch_size = 30, shuffle = True)
#
#     break
#
#     # model.train()
#     # running_loss = 0.0
#
#     # for i, (inputs, )
