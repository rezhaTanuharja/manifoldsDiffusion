import torch
import matplotlib.pyplot as plt

N = 128

wave_numbers = torch.tensor([i for i in range(N)])

time = 0.5
multipliers = torch.exp(-time * wave_numbers ** 2)
icoeffs = N * torch.ones(size = (N,))
icoeffs[0] = 0
# icoeffs[1:] = icoeffs[1:] * (torch.pi * wave_numbers[1:])
icoeffs = icoeffs * multipliers
rcoeffs = torch.zeros(size = icoeffs.shape)
# icoeffs = torch.ones(size = (64,))
icoeffs[0] = 0.0
# icoeffs[0] = 64 * 1.0
# results = torch.fft.rfft(icoeffs)
results = -torch.fft.irfft(torch.complex(rcoeffs, icoeffs * torch.pi * wave_numbers)) / torch.fft.irfft(torch.complex(icoeffs, rcoeffs))

plt.plot([i for i in range(len(results))], results)
# plt.ylim([-0.1, 1.1])
plt.show()
