

import torch
import numpy
import euleriandiffuser


def main():

    device = torch.device('cpu')
    rotations = numpy.random.random(size = (20, 3, 3))

    rotation_pipeline = euleriandiffuser.create_rotation_pipeline(
        device = device
    )

    noised_rotations = rotation_pipeline(rotations)

    print(noised_rotations)

if __name__ == "__main__":
    main()
