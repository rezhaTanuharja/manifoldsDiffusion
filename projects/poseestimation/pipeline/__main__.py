
import torch
from projects.poseestimation.dataloaders import tensorflowadaptor
from projects.poseestimation.pipeline.images import resnet
from projects.poseestimation.pipeline.rotations import euleriandiffuser
from projects.poseestimation.pipeline.times import sinusoidencoders
from projects.poseestimation.models.naive import NaiveMLP

import timeit


num_sample_duplicates = 1
num_timestamps = 5
device = torch.device('cuda')

try:

    image_pipeline = resnet.create_image_pipeline(
        num_sample_duplicates = num_sample_duplicates,
        num_timestamps = num_timestamps,
        device = device
    )

    label_pipeline = euleriandiffuser.create_rotation_pipeline(
        num_sample_duplicates = num_sample_duplicates,
        num_timestamps = num_timestamps,
        device = device
    )

    model = NaiveMLP(
        num_image_features = 1000,
        num_time_features = 8
    )

    model = model.to(device)

except Exception as e:

    print(f"Failed to generate a NumPy iterator: {type(e)}")
    raise


def main(rank: int = 0, world_size: int = 1):

    split_start = rank / world_size * 85
    split_end = (rank + 1) / world_size * 85

    split = f'train[{split_start}%:{split_end}%]'

    dataset = {
        'name': 'symmetric_solids',
        'split': split,
        'as_supervised': True,
        'shuffle_files': True,
    }

    batch_size = 20
    num_sample_duplicates = 1
    # num_timestamps = 5
    device = torch.device('cuda')

    try:

        data_iterator, _ = tensorflowadaptor.create_numpy_iterator(
            dataset = dataset,
            batch_size = batch_size,
        )

        times_pipeline = sinusoidencoders.create_time_pipeline(
            num_samples = batch_size,
            num_sample_duplicates = num_sample_duplicates,
            num_wave_numbers = 8,
            device = device
        )

        model = NaiveMLP(
            num_image_features = 1000,
            num_time_features = 8
        )

        model = model.to(device)

    except Exception as e:

        print(f"Failed to generate a NumPy iterator: {type(e)}")
        raise

    for _ in range(1000):

        images, labels = next(data_iterator)

        images = image_pipeline(images)
        labels = label_pipeline(labels)


        rotations = labels['rotations']
        times = times_pipeline(labels['time'])

        output = model(images, times, rotations)

        print(output.shape)

        break


    # print(output.shape)


if __name__ == "__main__":

    local_rank = 0
    world_size = 1

    # main()

    execution_time = timeit.timeit(
        main,
        number = 5
    )

    print(f"Execution time is {execution_time} seconds")
