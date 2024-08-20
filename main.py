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
forward_SDE = StandardOU(speed = 0.2, volatility = 1.0)


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
    time_increment = time_increment
)

# -- Postprocess and plot
plt.plot(time_stamps.cpu(), noisy_data.cpu().view(num_samples, dimension))
plt.show()

# for i in range(N):
#
#     X = time_integrator.step_forward(forward_SDE, X, t_series[i], dt)
#     data_recorder.store(X)
#     # print(result)
#     # print(X_series[0, i+1, 0])



# Y_series = torch.zeros(N)
# Y_series[0] = X_series[-1]
# Y_ref = X_series[0]
#
# drift_corrector = DirectToReference(Y_ref, t_series[-1])
# reverse_SDE = CorrectedNegative(forward_SDE, drift_corrector)
#
# for i in range(N-1):
#
#     Y_series[i + 1] = time_integrator.step_forward(reverse_SDE, Y_series[i], t_series[i], dt)

# X_series = data_recorder.get_record()
# print(X_series.type)

