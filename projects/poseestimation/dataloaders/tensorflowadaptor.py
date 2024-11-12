"""
poseestimation.dataloaders.tensorflowadaptor
============================================

Provides function to load TensorFlow datasets as a NumPy array generator

Functions
---------
`create_numpy_iterator` : create a NumPy iterator for a TensorFlow dataset
"""


import tensorflow
import tensorflow_datasets
import numpy

import diffusionmodels.dataprocessing as dataprocessing

from typing import cast, Dict, Any, Iterator, List, Tuple


def create_numpy_iterator(
    dataset: Dict[str, Any],
    batch_size: int = 1,
) -> Tuple[Iterator[List[numpy.ndarray]], int]:

    dataset_pipeline = dataprocessing.Pipeline(
        transforms = [

            # roll the dataset to have an "infinite" length
            lambda dataset: dataset.repeat(),

            # shuffle data
            # lambda dataset: dataset.shuffle(buffer_size = 4 * batch_size),
            lambda dataset: dataset.shuffle(buffer_size = batch_size),

            # retrieve a batch of data at a time
            lambda dataset: dataset.batch(batch_size),

            # convert dataset into a NumPy iterator
            lambda dataset: dataset.as_numpy_iterator(),

            # mainly for type checking
            lambda dataset: cast(Iterator[List[numpy.ndarray]], dataset),

        ]
    )

    try:

        tensorflow_data = tensorflow_datasets.load(**dataset)
        tensorflow_data = cast(tensorflow.data.Dataset, tensorflow_data)

    except Exception as e:

        print(f"Failed to load tensorflow_dataset: {type(e)}")
        raise

    iterator_length = len(tensorflow_datasets.as_numpy(tensorflow_data)) // batch_size

    return dataset_pipeline(tensorflow_data), iterator_length
