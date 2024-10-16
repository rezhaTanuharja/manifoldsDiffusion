import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import timeit
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TensorFlowDataWrapper(Dataset):

    def __init__(self, tensorflow_dataset):
        self._dataset = tensorflow_dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):

        image, rotation = next(iter(self._dataset))

        return tfds.as_numpy(image), tfds.as_numpy(rotation)


class CustomSampler(Sampler):

    def __init__(self, data_source):
        self._data_source = data_source

    def __iter__(self):
        indices = list(range(len(self._data_source)))
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self._data_source)


tensorflow_dataset = tfds.load('symmetric_solids', split = 'train', shuffle_files = True, as_supervised = True)
tensorflow_dataset = tensorflow_dataset.shuffle(buffer_size = 256).batch(32)


wrapped_dataset = TensorFlowDataWrapper(tensorflow_dataset)
custom_sampler = CustomSampler(wrapped_dataset)

data_loader = DataLoader(dataset = wrapped_dataset, batch_size = 1, shuffle = False)

def main():

    i = 0

    for image, label in data_loader:
    # for image, label in tensorflow_dataset:

        if i > 256:
            break

