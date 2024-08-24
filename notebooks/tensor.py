import numpy as np
import torch

from diffusionmodels.manifolds import SpecialOrthogonal3
from diffusionmodels.differentialequations import ExplodingRotationVariance, CorrectedNegative
from diffusionmodels.timeintegrators import EulerMaruyama
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler
from diffusionmodels.scorefunctions import DirectToReference


file_name = 'downloads/dmpl_sample.npz'


data = np.load(file_name)['poses']
data = data[0].reshape(1, 52, 3)

data = torch.tensor(data, device = "cuda").float()

iden = torch.eye(3, device = "cuda").reshape(1, 1, 3, 3).expand(1, 52, 3, 3)
print(iden.shape)

# print(data.shape)
# print(data.type)

manifold = SpecialOrthogonal3()
SDE = ExplodingRotationVariance(manifold)
time_integrator = EulerMaruyama()

sampler = SimpleSampler(time_integrator, SimpleRecorder())


rotation = manifold.exp(iden, data)
score_function = DirectToReference(manifold, rotation, 100 * 0.003)

noisy_rotation = sampler.get_samples(
    sde = SDE,
    initial_condition = rotation,
    num_samples = 100,
    dt = 0.003
)

final_condition = noisy_rotation[-1]

reverse_SDE = CorrectedNegative(SDE, score_function)

denoised_rotation = sampler.get_samples(
    sde = reverse_SDE,
    initial_condition = final_condition,
    num_samples = 100,
    dt = 0.003
)

iden = torch.eye(3, device = "cuda").reshape(1, 1, 1, 3, 3).expand(100, 1, 52, 3, 3)

to_save = manifold.log(iden, denoised_rotation)
to_save = to_save.reshape(100, 156)

numpy_array = to_save.cpu().numpy()
np.savez('downloads/postprocessed.npz', poses=numpy_array)


# diffusion = SDE.diffusion(rotation, 0.0)
# rotation = manifold.exp(iden, diffusion)
#
# angles = torch.acos(0.5 * (
#         torch.einsum('...ii -> ...', rotation) - 1.0
#     )
# )


# print(diffusion.shape)
print(to_save.shape)
