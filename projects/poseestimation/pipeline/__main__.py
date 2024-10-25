
import torch
from projects.poseestimation.dataloaders import tensorflowadaptor
from projects.poseestimation.pipeline.images import resnet
from projects.poseestimation.pipeline.rotations import euleriandiffuser
from projects.poseestimation.pipeline.times import sinusoidencoders


def main(rank: int, world_size: int):

    dataset = {
        'name': 'symmetric_solids',
        'split': 'train',
        'as_supervised': True,
        'shuffle_files': True,
    }

    num_sample_duplicates = 5
    num_timestamps = 3
    device = torch.device('cpu')

    try:

        dataloader = tensorflowadaptor.create_local_numpy_iterator(
            dataset = dataset,
            batch_size = 20,
            rank = rank,
            world_size = world_size,
        )

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

        times_pipeline = sinusoidencoders.create_time_pipeline(
            num_samples = 20,
            num_sample_duplicates = num_sample_duplicates,
            device = device
        )

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
