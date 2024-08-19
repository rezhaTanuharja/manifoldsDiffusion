import matplotlib.pyplot as plt
import torch

from diffusion_models.differential_equations import standard_OU
from diffusion_models.time_integrators import Euler_Maruyama


mySDE = standard_OU(speed=0.5, volatility=0.005)
myIntegrator = Euler_Maruyama()


N = 5000
dt = 0.001

t_series = dt * torch.arange(N)
X_series = torch.zeros(N)

X_series[0] = 10.0


for i in range(N-1):

    # print(i)
    X_series[i + 1] = myIntegrator.step_forward(mySDE, X_series[i], t_series[i], dt)


plt.plot(t_series, X_series)
plt.show()
