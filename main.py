import matplotlib.pyplot as plt
import torch

from diffusionmodels.differentialequations import StandardOU, CorrectedNegative
from diffusionmodels.timeintegrators import EulerMaruyama, Heun
from diffusionmodels.scorefunctions import DirectToReference


N = 5000
dt = 0.001

t_series = dt * torch.arange(N)
X_series = torch.zeros(N)

X_series[0] = 10.0

forward_SDE = StandardOU(speed = 0.5, volatility = 2.0)
time_integrator = Heun(EulerMaruyama())

for i in range(N-1):

    X_series[i + 1] = time_integrator.step_forward(forward_SDE, X_series[i], t_series[i], dt)


Y_series = torch.zeros(N)
Y_series[0] = X_series[-1]
Y_ref = X_series[0]

drift_corrector = DirectToReference(Y_ref, t_series[-1])
reverse_SDE = CorrectedNegative(forward_SDE, drift_corrector)

for i in range(N-1):

    Y_series[i + 1] = time_integrator.step_forward(reverse_SDE, Y_series[i], t_series[i], dt)


plt.plot(t_series, X_series)
plt.plot(t_series, Y_series.flip(dims=[0]))
plt.show()
