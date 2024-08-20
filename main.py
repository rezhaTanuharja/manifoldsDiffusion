import matplotlib.pyplot as plt
import torch

from diffusionmodels.differentialequations import StandardOU, CorrectedNegative
from diffusionmodels.timeintegrators import EulerMaruyama, Heun
from diffusionmodels.scorefunctions import DirectToReference
from diffusionmodels.samplers import SimpleRecorder, SimpleSampler


N = 5000
dim = 5
dt = 0.001

t_series = dt * torch.arange(N)
# X_series = torch.zeros(1, N, 1)

X_init = 10.0 * torch.ones(1, dim)
X = X_init
# X_series[0, 0, 0] = 10.0

data_recorder = SimpleRecorder()
data_recorder.reset(X_init, N)

forward_SDE = StandardOU(speed = 0.0, volatility = 0.1)
time_integrator = Heun(EulerMaruyama())

solution_sampler = SimpleSampler(time_integrator, data_recorder)


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

X_series = solution_sampler.get_samples(forward_SDE, X, N, dt)
# X_series = data_recorder.get_record()
# print(X_series.type)

plt.plot(t_series, X_series.view(N, dim))
# # # plt.plot(t_series, Y_series.flip(dims=[0]))
plt.show()
