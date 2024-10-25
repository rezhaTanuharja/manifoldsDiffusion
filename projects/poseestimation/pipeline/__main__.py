
import torch
from typing import cast, Iterator
from projects.poseestimation.dataloaders import tensorflowadaptor
from projects.poseestimation.pipeline.images import resnet
from projects.poseestimation.pipeline.rotations import euleriandiffuser
from projects.poseestimation.pipeline.times import sinusoidencoders
import numpy


def main(rank: int, world_size: int):

    dataset = {
        'name': 'symmetric_solids',
        'split': 'train',
        'as_supervised': True,
        'shuffle_files': True,
    }

    device = torch.device('cpu')

    try:

        dataloader = tensorflowadaptor.create_local_numpy_iterator(
            dataset = dataset,
            batch_size = 20,
            rank = rank,
            world_size = world_size,
        )

        dataloader = cast(Iterator[numpy.ndarray], dataloader)

        image_pipeline = resnet.create_image_pipeline(device = device)
        label_pipeline = euleriandiffuser.create_rotation_pipeline(device = device)
        times_pipeline = sinusoidencoders.create_time_pipeline(device = device)

    except Exception as e:

        print(f"Failed to generate a NumPy iterator: {type(e)}")
        raise

    images, labels = next(dataloader)

    images = image_pipeline(images)
    labels = label_pipeline(labels)


    times = times_pipeline(labels['time'])


    print(images.shape)
    # print(labels.shape)
    print(times.shape)

if __name__ == "__main__":

    local_rank = 0
    world_size = 1

    main(rank = local_rank, world_size = world_size)
