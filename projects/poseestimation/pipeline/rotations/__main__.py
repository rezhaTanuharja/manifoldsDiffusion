

import torch
import numpy
import euleriandiffuser


def main():

    device = torch.device('cpu')
    rotations = numpy.random.random(size = (20, 3, 3))

    rotation_pipeline = euleriandiffuser.create_rotation_pipeline(
        num_sample_duplicates = 5,
        num_timestamps = 3,
        device = device
    )

    noised_rotations = rotation_pipeline(rotations)

    print(noised_rotations)

if __name__ == "__main__":
    main()
