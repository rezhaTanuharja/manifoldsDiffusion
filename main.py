import matplotlib.pyplot as plt
import torch

from diffusionmodels.differentialequations import StandardOU, CorrectedNegative
from diffusionmodels.timeintegrators import EulerMaruyama, Heun
from diffusionmodels.scorefunctions import DirectToReference
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler


# -- Use NVIDIA GPU whenever available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Create a sample data
num_subjects = 1
dimension = 8

X = 10.0 * torch.ones(num_subjects, dimension, device = device)

# -- Define a stochastic differential equation
forward_SDE = StandardOU(speed = 0.0, volatility = 2.0)


# -- Define parameters for sampling
num_samples = 20000
time_increment = 0.001

time_stamps = time_increment * torch.arange(num_samples, device = device)

# -- Create a sampler
time_integrator = Heun(EulerMaruyama())
data_recorder = SimpleRecorder()

solution_sampler = SimpleSampler(
    time_integrator = time_integrator,
    data_recorder = data_recorder
)

# -- Solve SDE and store solutions
noisy_data = solution_sampler.get_samples(
    sde = forward_SDE,
    initial_condition = X,
    num_samples = num_samples,
    dt = time_increment
)

# -- Prepare denoising steps
Y = noisy_data[-1]
drift_corrector = DirectToReference(noisy_data[0], time_stamps[-1])
reverse_SDE = CorrectedNegative(forward_SDE, drift_corrector)

# -- Sample from the reversed SDE
denoised_data = solution_sampler.get_samples(
    sde = reverse_SDE,
    initial_condition = Y,
    num_samples = num_samples,
    dt = time_increment
)

# -- Postprocess and plot
plt.plot(time_stamps.cpu(), noisy_data.cpu().view(num_samples, dimension), color = 'red', alpha = 0.2)
plt.plot((time_stamps[-1] + time_stamps).cpu(), denoised_data.cpu().view(num_samples, dimension), color = 'green', alpha = 0.2)
plt.show()
