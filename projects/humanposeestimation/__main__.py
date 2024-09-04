import yaml
import torch
import numpy as np
import diffusionmodels as dm

from torch.utils.data import DataLoader
from models.definitions import NaiveMLP


# -- load the simulation parameters
with open('./projects/humanposeestimation/parameters.yaml') as config_file:
    param = yaml.safe_load(config_file)

# -- use NVIDIA GPU if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### NOTE: 
# -- This part defines the diffusion problem:
# --    manifold
# --    stochastic differential equations 
# --    SDE solver
# --    score function

manifold = dm.manifolds.SpecialOrthogonal3()

stochastic_de = dm.differentialequations.ExplodingVariance(

    manifold = manifold,
    variance_scale = param['time_increment'] ** 0.5

)

solver = dm.solvers.SimpleSolver(
    time_integrator = dm.timeintegrators.EulerMaruyama(),
    data_recorder = dm.recorders.SimpleRecorder(),
    num_points = param['num_time_points'],
    grid_size = param['time_increment']
)

score_function = dm.scorefunctions.Direction(manifold = manifold)

# -- send all objects to the same device
for obj in [
    manifold,
    stochastic_de,
    solver,
    score_function,
]:
    obj.to(device)


### NOTE: 
# -- This part defines the data processing steps

data_pipeline = dm.dataprocessing.Pipeline(
    transforms = [

        # -- move dataset to the same device
        lambda dataset: dataset.to(device),

        # -- AMASS dataset is a 2D tensor with shape (num_subjects, num_joints * tangent_dimension)
        # -- reshape to (num_subjects, num_points, *manifold.tangent_dimension)
        lambda dataset: dataset.unflatten(-1, (-1, *manifold.tangent_dimension())),

        # -- convert each tangent vector into its corresponding rotation matrix
        # -- return a tensor with shape (num_subjects, num_joints, *manifold.dimension)
        lambda dataset: manifold.exp(
            torch.eye(3, device = device).view(*manifold.dimension()),
            dataset
        ),

        # -- duplicate all subjects so we generate multiple noise paths simultaneously
        lambda dataset: dataset.unsqueeze(0).expand(
            param['num_subject_duplicates'], *dataset.shape
        ).flatten(0,1),

        # -- the forward noising process
        # -- returns {time, data} where data[i] is the noised data at time[i]
        lambda dataset: {
            **(solver.solve(dataset, stochastic_de)),
            'original': dataset
        },

        # -- use score function as data label for training
        lambda dataset: {

            'time': dataset['time'],
            'points': dataset['data'],

            'labels': score_function.get_direction(
                origin = dataset['data'],
                destination = dataset['original'],
                scale = 1.0 / dataset['time']
            )

        },

        # -- reshape tensors so the first shape is (num_subjects * num_subject_duplicates * num_time_points)
        # -- and the second one is the rest of the dimensions flattened into one
        lambda dataset: {

            'time': dataset['time'].reshape(
                *dataset['time'].shape, 1
            ).expand(
                -1, param['num_subject_duplicates'] * param['subject_batch']
            ).flatten(0, 1).unsqueeze(-1),

            'points': dataset['points'].flatten(0, 1).flatten(1),
            'labels': dataset['labels'].flatten(0, 1).flatten(1)

        },

    ]
)


### NOTE: 
# -- This part defines the NN model, loss function, and optimizer

model = NaiveMLP()
model.load_state_dict(torch.load('projects/humanposeestimation/models/naive_model.pth'))
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = param['learning_rate'])

datasets = np.load(param['file_name'])['poses']
datasets = torch.tensor(datasets[:param['num_subjects']]).float()

data_loader = DataLoader(datasets, batch_size = param['subject_batch'], shuffle = True)


### NOTE: 
# -- This is the model training process

for i in range(param['num_epochs']):

    model.train()
    running_loss = 0.0

    for dataset in data_loader:

        # -- process data using the predefined pipeline
        data = data_pipeline(dataset)

        # -- forward
        optimizer.zero_grad()
        outputs = model(data['points'], data['time'])

        # -- back propagation
        loss = criterion(outputs, data['labels'])
        loss.backward()

        # -- avoid huge update by clipping the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = param['max_norm'])

        # -- model update
        optimizer.step()
        running_loss += loss.item()

    # -- periodic save to avoid losing huge progress
    if i % 250 == 0:
        torch.save(model.state_dict(), 'projects/humanposeestimation/models/naive_model.pth')

    # -- output training progress
    print(f'Epoch [{(i + 1):04}/{param["num_epochs"]}], Loss: {(running_loss / len(data_loader)):.4f}')

