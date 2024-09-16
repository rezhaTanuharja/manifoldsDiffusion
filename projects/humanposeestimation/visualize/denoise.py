import yaml
import torch

import numpy as np
import diffusionmodels as dm
from models.definitions import NaiveMLP

def main():

    # -- load parameters from file
    with open('./projects/humanposeestimation/parameters.yaml') as config_file:
        param = yaml.safe_load(config_file)

    # -- we don't always need cuda for this, so only use when available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    manifold = dm.manifolds.SpecialOrthogonal3()

    solver = dm.solvers.SimpleSolver(
        time_integrator = dm.timeintegrators.EulerMaruyama(),
        data_recorder = dm.recorders.SimpleRecorder(),
        num_points = param['num_time_points'] * 8,
        grid_size = param['time_increment']
    )

    state_dict = torch.load('projects/humanposeestimation/models/naive_scaled_model.pth')
    state_dict ={
        key.replace('module.', ''): value for key, value in state_dict.items()
    }
    model = NaiveMLP()
    model.load_state_dict(state_dict)
    model.to(device)

    stochastic_de = dm.differentialequations.ExplodingVariance(

        manifold = manifold,
        variance_scale = 0.5

    )

    reversed_stochastic_de = dm.differentialequations.CorrectedNegative(
        stochastic_de = stochastic_de,
        drift_corrector = model
    )

    for obj in (
        manifold,
        stochastic_de,
        reversed_stochastic_de,
        solver,
    ):
        obj.to(device)

    data_pipeline = dm.dataprocessing.Pipeline(
        transforms = [

            # -- move dataset to the same device
            lambda dataset: dataset.to(device, non_blocking = True),

            # -- AMASS dataset is a 2D tensor with shape (num_subjects, num_joints * tangent_dimension)
            # -- reshape to (num_subjects, num_points, *manifold.tangent_dimension)
            lambda dataset: dataset.unflatten(-1, (-1, *manifold.tangent_dimension())),

            # -- convert each tangent vector into its corresponding rotation matrix
            # -- return a tensor with shape (num_subjects, num_joints, *manifold.dimension)
            lambda dataset: manifold.exp(
                torch.eye(3, device = device).view(*manifold.dimension()),
                dataset
            ),

            # -- the forward noising process
            # -- returns {time, data} where data[i] is the noised data at time[i]
            lambda dataset: (solver.solve(dataset, reversed_stochastic_de))['data'],

            lambda dataset: dataset.flatten(0, 1),

            lambda dataset: manifold.log(
                torch.eye(3, device = device).view(*manifold.dimension()),
                dataset
            ),

            lambda dataset: dataset.flatten(1)

        ]
    )

    datasets = np.load('./projects/humanposeestimation/visualize/data/A2_subject1_noised_0.npz')['poses']
    datasets = torch.tensor(datasets[1000:1001]).float()

    datasets = data_pipeline(datasets)
    print(datasets.shape)
    datasets = datasets.detach().cpu().numpy()
    np.savez('./projects/humanposeestimation/visualize/data/A2_subject1_denoised_0.npz', poses=datasets)

if __name__ == "__main__":
    main()
