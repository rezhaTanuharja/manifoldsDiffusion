
import torch
from projects.poseestimation.dataloaders import tensorflowadaptor
from projects.poseestimation.pipeline.images import resnet
from projects.poseestimation.pipeline.rotations import euleriandiffuser
from projects.poseestimation.pipeline.times import sinusoidencoders
from projects.poseestimation.models.naive import NaiveMLP


def main(rank: int, world_size: int):

    dataset = {
        'name': 'symmetric_solids',
        'split': 'train',
        'as_supervised': True,
        'shuffle_files': True,
    }

    batch_size = 4
    num_sample_duplicates = 2
    num_timestamps = 2
    device = torch.device('cuda')

    try:

        dataloader = tensorflowadaptor.create_local_numpy_iterator(
            dataset = dataset,
            batch_size = batch_size,
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
            num_samples = batch_size,
            num_sample_duplicates = num_sample_duplicates,
            num_wave_numbers = 4,
            device = device
        )

        model = NaiveMLP(
            num_image_features = 1000,
            num_time_features = 4
        )

        model = model.to(device)

    except Exception as e:

        print(f"Failed to generate a NumPy iterator: {type(e)}")
        raise

    for images, labels in dataloader:

        images = image_pipeline(images)
        labels = label_pipeline(labels)


        rotations = labels['rotations']
        times = times_pipeline(labels['time'])

        output = model(images, times, rotations)

        break


    print(output.shape)

if __name__ == "__main__":

    local_rank = 0
    world_size = 1

    main(rank = local_rank, world_size = world_size)
