import torch
from torch.utils.data import DataLoader, TensorDataset

import diffusionmodels as dm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_samples = 1000
time_increment = 0.001

# -- the stochastic differential equation is an exploding variance sde in SO3
stochastic_de = dm.differentialequations.ExplodingVariance(

    manifold = dm.manifolds.SpecialOrthogonal3(device = device),
    variance_scale = time_increment ** 0.5

)

solver = dm.solvers.SimpleSolver(
    time_integrator = dm.timeintegrators.EulerMaruyama(),
    data_recorder = dm.recorders.SimpleRecorder(),
    num_points = num_samples,
    grid_size = time_increment
)

transform_pipeline = dm.pipeline.Pipeline([

    # -- move to device
    (lambda datasets: [dataset.to(device) for dataset in datasets]),

    # -- pair with their respective sde
    (lambda datasets: list(zip(datasets, sdes))),

    # -- convert tangent vectors to points
    dm.utilities.extract_points_from_amass,
    # fix_problems,

    # -- forward noising process
    solver.solve,

    # -- flatten dimension
    (lambda datasets: [dataset.flatten(0,1).flatten(1) for dataset in datasets])
])

data = [
    torch.randn(10, 156),
    torch.randn(10, 156),
]

sdes = [
    stochastic_de,
    stochastic_de,
]




dataset = TensorDataset(*data)
dloader = DataLoader(dataset, batch_size = 2, shuffle = True)


for i, data in enumerate(dloader):

    noised_rotations = transform_pipeline(data)
