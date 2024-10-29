
import torch
from torch.nn import MSELoss
from torch.optim.adam import Adam

from projects.poseestimation.dataloaders import tensorflowadaptor
from projects.poseestimation.pipeline.images import resnet
from projects.poseestimation.pipeline.rotations import euleriandiffuser
from projects.poseestimation.pipeline.times import sinusoidencoders
from projects.poseestimation.models.naive import NaiveMLP


def main(rank: int, world_size:int):


    device = torch.device('cuda')

    dataset = {
        'name': 'symmetric_solids',
        'split': 'train',
        'as_supervised': True,
        'shuffle_files': True,
    }

    batch_size = 20
    num_sample_duplicates = 1
    num_timestamps = 5
    num_wave_numbers = 8

    num_epochs = 5000


    checkpoint_file = 'checkpoint.pth'

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

        times_pipeline = sinusoidencoders.create_time_pipeline(
            num_samples = batch_size,
            num_sample_duplicates = num_sample_duplicates,
            num_wave_numbers = num_wave_numbers,
            device = device
        )

        model = NaiveMLP(
            num_image_features = 1000,
            num_time_features = num_wave_numbers
        )

        model = model.to(device)

        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr = 0.001)

    except Exception as e:

        print(f"Failed to create pipelines: {type(e)}")
        raise



    for epoch in range(num_epochs):

        running_loss = 0.0
        num_batches = 0


        try:
            data_iterator, iterator_length = tensorflowadaptor.create_local_numpy_iterator(
                dataset = dataset,
                batch_size = batch_size,
                rank = rank,
                world_size = world_size,
            )

        except Exception as e:

            print(f"Failed to create a local NumPy iterator: {type(e)}")
            raise


        for _ in range(iterator_length):

            images, labels = next(data_iterator)

            images = image_pipeline(images)
            labels = label_pipeline(labels)

            rotations = labels['rotations']
            times = times_pipeline(labels['time'])
            velocities = labels['velocities']

            outputs = model(images, times, rotations)

            loss = criterion(outputs, velocities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if num_batches % 500 == 0:
                print(f"Processed {num_batches * batch_size * num_sample_duplicates} images from epoch {epoch}")


        print(f"Epoch {epoch}/{num_epochs}, loss: {running_loss/num_batches}")


        if epoch % 500 == 0:

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

            torch.save(checkpoint, checkpoint_file)


if __name__ == "__main__":

    local_rank = 0
    world_size = 1

    main(rank = local_rank, world_size = world_size)
