

import torch
import numpy
import sinusoidencoders


def main():

    device = torch.device('cpu')
    times = torch.tensor(
        numpy.random.random(size = (20,))
    )

    time_pipeline = sinusoidencoders.create_time_pipeline(
        num_samples = times.numel(),
        num_sample_duplicates = 5,
        device = device
    )

    output = time_pipeline(times)

    print(output)




if __name__ == "__main__":
    main()
