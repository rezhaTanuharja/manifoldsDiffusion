import torch.nn as nn

class NaiveMLP(nn.Module):

    def __init__(self):

        super(NaiveMLP, self).__init__()

        self.fc1 = nn.Linear(468, 512)
        self.fct = nn.Linear(  1, 512, bias = False)

        self.fc2 = nn.Linear(512, 512)

        self.fcf = nn.Linear(512, 156)

        self.leaky_ReLU = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x, t):

        x = self.leaky_ReLU(self.fc1(x) + self.fct(t))

        x = self.leaky_ReLU(self.fc2(x))

        x = self.fcf(x)
        return x
