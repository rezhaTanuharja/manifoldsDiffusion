"""
poseestimation.dataloaders.__main__
===================================

Mainly used for debugging the dataloaders
"""

#WARN: torch.distributed MUST be loaded before loading any other modules
import torch.distributed

from typing import cast, Iterator
import tensorflowadaptor
import numpy


import os
import debugpy

debug = os.getenv("DEBUG_FLAG", "0")

if debug == "1":
    rank = int(os.getenv("RANK", "-1"))
    port = rank + 5678
    debugpy.listen(("127.0.0.1", port))
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main(rank: int, world_size: int) -> None:

    dataset = {
        'name': 'symmetric_solids',
        'split': 'train',
        'as_supervised': True,
        'shuffle_files': True,
    }

    try:

        dataloader = tensorflowadaptor.create_local_numpy_iterator(
            dataset = dataset,
            batch_size = 20,
            rank = rank,
            world_size = world_size,
        )

        dataloader = cast(Iterator[numpy.ndarray], dataloader)

    except Exception as e:

        print(f"Failed to generate a NumPy iterator: {type(e)}")
        raise

    images, labels = next(dataloader)

    print(images.shape)
    print(labels.shape)


if __name__ == "__main__":

    local_rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    torch.distributed.init_process_group(
        backend = "gloo",
        rank = local_rank,
        world_size = world_size
    )

    try:
        main(rank = local_rank, world_size = world_size)

    except Exception as e:
        print(f"Failed to execute main: {type(e)}")

    torch.distributed.destroy_process_group()
