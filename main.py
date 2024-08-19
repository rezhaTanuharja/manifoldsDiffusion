import matplotlib.pyplot as plt
import torch

from diffusion_models.differential_equations import standard_OU, corrected_negative
from diffusion_models.time_integrators import Euler_Maruyama, Heun
from diffusion_models.score_functions import direct_to_reference


N = 5000
dt = 0.001

t_series = dt * torch.arange(N)
X_series = torch.zeros(N)

X_series[0] = 10.0

forward_SDE = standard_OU(speed = 0.5, volatility = 0.05)
# time_integrator = Heun(Euler_Maruyama())
time_integrator = Euler_Maruyama()

for i in range(N-1):

    X_series[i + 1] = time_integrator.step_forward(forward_SDE, X_series[i], t_series[i], dt)


Y_series = torch.zeros(N)
Y_series[0] = X_series[-1]
Y_ref = X_series[0]

drift_corrector = direct_to_reference(Y_ref, t_series[-1])
reverse_SDE = corrected_negative(forward_SDE, drift_corrector)

for i in range(N-1):

    Y_series[i + 1] = time_integrator.step_forward(reverse_SDE, Y_series[i], t_series[i], dt)


plt.plot(t_series, X_series)
plt.plot(t_series, Y_series)
plt.show()
