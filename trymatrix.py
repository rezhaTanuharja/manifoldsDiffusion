from diffusionmodels.manifolds import SO3

import torch


manifold = SO3()

X = torch.zeros(2, 3, 3, 3)
X[0, 0] = torch.eye(3)
X[0, 1] = torch.eye(3)
X[0, 2] = torch.eye(3)
X[1, 0] = torch.eye(3)
X[1, 1] = torch.eye(3)
X[1, 2] = torch.eye(3)

dX = torch.pi / 3.0 * torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])


Y = manifold.exp(X, dX)

theta = manifold.log(torch.eye(3), Y)

print(Y)
print(theta)
