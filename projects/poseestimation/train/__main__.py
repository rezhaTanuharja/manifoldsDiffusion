import os
import torch

import dataset
import pipeline

import debugpy

debug = os.getenv("DEBUG_FLAG", "0")

if debug == "1":
    rank = int(os.getenv("RANK", "-1"))
    if rank == 0:
        debugpy.listen(("127.0.0.1", 5678))
        debugpy.wait_for_client()
        debugpy.breakpoint()

def main(rank: int, world_size:int):

    torch.distributed.init_process_group(
        backend = "gloo",
        rank = rank,
        world_size = world_size
    )
    
    data_generator = dataset.create_local_generator(
        batch_size = 32,
        rank = rank,
        world_size = world_size
    )

    for _ in range(1):

        image, label = next(data_generator)

        image = pipeline.image_pipeline(image)
        label = pipeline.label_pipeline(label)

        #TODO: the training part
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":

    local_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    main(rank = local_rank, world_size = world_size)
