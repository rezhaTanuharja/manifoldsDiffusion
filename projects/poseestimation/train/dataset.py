import tensorflow_datasets as tfds
import tensorflow as tf
from typing import cast
import torch.distributed as dist
import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_local_generator(batch_size: int, rank: int = 0, world_size: int = 1):

    if rank == 0:
        global_seed = torch.randint(low = 0, high = 1000000, size = (1,), dtype = torch.int)
    else:
        global_seed = torch.zeros(size = (1,))

    dist.broadcast(tensor = global_seed, src = 0)

    tensorflow_read_config = tfds.ReadConfig(shuffle_seed = int(global_seed.item()))

    tensorflow_data = tfds.load(
        name = 'symmetric_solids',
        split = 'train',
        as_supervised = True,
        shuffle_files = True,
        read_config = tensorflow_read_config
    )

    try:
        tensorflow_data = cast(tf.data.Dataset, tensorflow_data)
    except TypeError as e:
        print(f"Casting failed: {e}")
        raise

    dist.barrier()
    chunk_size = len(tfds.as_numpy(tensorflow_data)) // world_size
    shuffle_buffer_size = 4 * batch_size

    tensorflow_data = tensorflow_data.skip(rank * chunk_size)
    tensorflow_data = tensorflow_data.take(chunk_size)
    tensorflow_data = tensorflow_data.repeat()
    tensorflow_data = tensorflow_data.shuffle(buffer_size = shuffle_buffer_size)
    tensorflow_data = tensorflow_data.batch(batch_size)
    tensorflow_data = tensorflow_data.as_numpy_iterator()

    return tensorflow_data
