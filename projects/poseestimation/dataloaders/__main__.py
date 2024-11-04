"""
poseestimation.dataloaders.__main__
===================================

Mainly used for debugging the dataloaders
"""

import tensorflowadaptor


def main(rank: int = 0, world_size: int = 1) -> None:

    # simulate validation = 'train[85%:]'

    split_start = rank / world_size * 85
    split_end = (rank + 1) / world_size * 85

    split = f'train[{split_start}%:{split_end}%]'

    dataset = {
        'name': 'symmetric_solids',
        'split': split,
        'as_supervised': True,
        'shuffle_files': True,
    }

    try:

        data_iterator, _ = tensorflowadaptor.create_numpy_iterator(
            dataset = dataset,
            batch_size = 20,
        )

    except Exception as e:

        print(f"Failed to generate a NumPy iterator: {type(e)}")
        raise

    images, labels = next(data_iterator)

    print(images.shape)
    print(labels.shape)


if __name__ == "__main__":

    # simulate parallel process

    local_rank = 2
    world_size = 4

    main(rank = local_rank, world_size = world_size)
