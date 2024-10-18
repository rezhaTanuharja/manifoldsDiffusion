"""
poseestimation.dataloaders.tensorflowadaptor
============================================

Provides function to load TensorFlow datasets as a NumPy array generator

Functions
---------
`create_local_numpy_iterator` : create a NumPy iterator from a TensorFlow dataset
"""


import tensorflow
import tensorflow_datasets
import torch
import torch.distributed
from typing import cast, Dict, Any


def generate_common_seed(local_rank: int) -> int:
    """
    Generate a random seed that is shared among all processes

    Parameters
    ----------
    `local_rank: int`
        The rank of the process that calls this function

    Returns
    -------
    `int`
        A random integer but has the same value across all processes
    """

    if local_rank == 0:
        global_seed = torch.randint(
            low = 0, high = 1000000,
            size = (1,),
            dtype = torch.int,
        )
    else:
        global_seed = torch.zeros(
            size = (1,),
            dtype = torch.int
        )

    try:

        torch.distributed.broadcast(tensor = global_seed, src = 0)

    except Exception as e:

        print(f"Failed to distribute a global seed: {type(e)}")
        raise

    return int(global_seed.item())



def create_local_numpy_iterator(
    dataset: Dict[str, Any],
    batch_size: int = 1,
    rank: int = 0,
    world_size: int = 1
):
    """
    Create a NumPy iterator to retrieve data from a TensorFlow dataset

    Parameters
    ----------
    `dataset: Dict[str, Any]`
        A dictionay with at least `name: "dataset_name"` and `split: "train/test"`

    `batch_size: int = 1`
        The number of data points to produce whenever the generator is called with next

    `rank: int = 0`
        The local rank of current process

    `world_size: int = 1`
        The world size, greater than one in a parallel computation

    Returns
    -------
    `Iterator[numpy.ndarray]`
        An iterator that, when called with next, produces a numpy array
    """

    if world_size == 1:

        try:

            tensorflow_data = tensorflow_datasets.load(**dataset)
            tensorflow_data = cast(tensorflow.data.Dataset, tensorflow_data)

        except Exception as e:

            print(f"Failed to load tensorflow_dataset: {type(e)}")
            raise

        tensorflow_data = tensorflow_data.repeat()
        tensorflow_data = tensorflow_data.shuffle(buffer_size = 4 * batch_size)
        tensorflow_data = tensorflow_data.batch(batch_size)
        tensorflow_data = tensorflow_data.as_numpy_iterator()

        return tensorflow_data

    tensorflow_read_config = tensorflow_datasets.ReadConfig(
        shuffle_seed = generate_common_seed(local_rank = rank)
    )

    try:

        tensorflow_data = tensorflow_datasets.load(
            **dataset,
            read_config = tensorflow_read_config
        )

        tensorflow_data = cast(tensorflow.data.Dataset, tensorflow_data)

    except Exception as e:

        print(f"Failed to load tensorflow_dataset: {type(e)}")
        raise

    chunk_size = len(tensorflow_datasets.as_numpy(tensorflow_data)) // world_size

    tensorflow_data = tensorflow_data.skip(rank * chunk_size)
    tensorflow_data = tensorflow_data.take(chunk_size)

    tensorflow_data = tensorflow_data.repeat()
    tensorflow_data = tensorflow_data.shuffle(buffer_size = 4 * batch_size)
    tensorflow_data = tensorflow_data.batch(batch_size)
    tensorflow_data = tensorflow_data.as_numpy_iterator()

    return tensorflow_data
