import torch

import diffusionmodels.initialvalueproblems as ivp
import diffusionmodels.solvers as slv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_increment = 0.001


# -- the stochastic differential equation is an exploding variance sde in SO3
stochastic_de = ivp.ExplodingVariance(
    manifold = ivp.SpecialOrthogonal3(device = device),
    variance_scale = time_increment ** 0.5
)

rotation = torch.eye(3, device = device)
rotation = rotation.view(1, 1, 3, 3).expand(1, 52, 3, 3)

revolution = torch.eye(3, device = device)
revolution = revolution.view(1, 1, 3, 3).expand(5, 22, 3, 3)

IVPs = ivp.InitialValueProblems()

IVPs.append(rotation, stochastic_de)
IVPs.append(revolution, stochastic_de)

# IVPs = (
#     {'initial_condition': rotation, 'stochastic_de': stochastic_de},
#     {'initial_condition': revolution, 'stochastic_de': stochastic_de},
# )

sampler = slv.SimpleSampler(
    time_integrator = slv.Heun(predictor = slv.EulerMaruyama()),
    data_recorder = slv.SimpleRecorder()
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
