import yaml
import torch

import numpy as np
import diffusionmodels as dm

def main():

    # -- load parameters from file
    with open('./projects/humanposeestimation/parameters.yaml') as config_file:
        param = yaml.safe_load(config_file)

    # -- we don't always need cuda for this, so only use when available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #NOTE: This part defines the forward noising process

    manifold = dm.manifolds.SpecialOrthogonal3()

    stochastic_de = dm.differentialequations.ExplodingVariance(

        manifold = manifold,
        variance_scale = 0.5

    )

    solver = dm.solvers.SimpleSolver(
        time_integrator = dm.timeintegrators.EulerMaruyama(),
        data_recorder = dm.recorders.SimpleRecorder(),
        num_points = param['num_time_points'] * 4,
        grid_size = param['time_increment']
    )

    for obj in (
        manifold,
        stochastic_de,
        solver,
    ):
        obj.to(device)

    #NOTE: This part defines the data-processing steps

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
            lambda dataset: (solver.solve(dataset, stochastic_de))['data'],

            lambda dataset: dataset.flatten(0, 1),

            lambda dataset: manifold.log(
                torch.eye(3, device = device).view(*manifold.dimension()),
                dataset
            ),

            lambda dataset: dataset.flatten(1)

        ]
    )

    datasets = np.load(param['file_name'])['poses']
    datasets = torch.tensor(datasets[0:1]).float()

    datasets = data_pipeline(datasets)

    datasets = datasets.detach().cpu().numpy()
    np.savez('./projects/humanposeestimation/visualize/data/A2_subject1_noised_0.npz', poses=datasets)

if __name__ == "__main__":
    main()
