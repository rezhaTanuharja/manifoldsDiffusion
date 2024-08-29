import torch

import diffusionmodels as dm

# -- aliases for conciseness
SpecialOrthogonal3 = dm.manifolds.SpecialOrthogonal3
ExplodingVariance = dm.differentialequations.ExplodingVariance
InitialValueProblems = dm.differentialequations.InitialValueProblems
SimpleSampler = dm.samplers.SimpleSampler
SimpleRecorder = dm.samplers.SimpleRecorder
EulerMaruyama = dm.timeintegrators.EulerMaruyama
Heun = dm.timeintegrators.Heun


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_increment = 0.001


# -- the stochastic differential equation is an exploding variance sde in SO3
stochastic_de = ExplodingVariance(
    manifold = SpecialOrthogonal3(device = device),
    variance_scale = time_increment ** 0.5
)

rotation = torch.eye(3, device = device)
rotation = rotation.view(1, 1, 3, 3).expand(1, 52, 3, 3)

revolution = torch.eye(3, device = device)
revolution = revolution.view(1, 1, 3, 3).expand(5, 22, 3, 3)

IVPs = InitialValueProblems()

IVPs.append(rotation, stochastic_de)
IVPs.append(revolution, stochastic_de)

sampler = SimpleSampler(
    time_integrator = Heun(predictor = EulerMaruyama()),
    data_recorder = SimpleRecorder()
)

noised_rotation = sampler.get_samples(
    # sde = stochastic_de,
    # initial_condition = rotation,
    IVPs,
    num_samples = 3,
    dt = 0.001
)

print(noised_rotation[0].shape)
print(noised_rotation[1].shape)
