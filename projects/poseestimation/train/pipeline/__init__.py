import tensorflow_datasets as tfds
import diffusionmodels as dm
import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pipeline.image as image_pipe
import pipeline.label as label_pipe


time = torch.arange(start = 0.0, end = 2.5, step = 0.5)
time = time.to(torch.device('cuda'))

num_samples = 32


image_pipeline = dm.dataprocessing.Pipeline(
    transforms = [

        # convert TF tensor to Torch tensor with a compatible data type
        lambda images: torch.tensor(images, dtype = torch.float),

        # move tensor to GPU
        lambda images: images.to(torch.device('cuda')),

        # in TF sample shape is 244, 244, 3, in Torch it should be 3, 244, 244
        lambda images: images.permute(0, 3, 1, 2),

        # parse image using pretrained model
        lambda images: image_pipe.model(images),

        # duplicate samples to noise each sample multiple times
        lambda images: images.unsqueeze(0).expand(
            5, *images.shape
        ).flatten(0, 1)

    ]
)

label_pipeline = dm.dataprocessing.Pipeline(
    transforms = [

        # convert TF tensor to Torch tensor
        lambda rotations: torch.tensor(rotations),

        # move tensor to GPU
        lambda rotations: rotations.to(torch.device('cuda')),

        # duplicate samples to noise each sample multiple times
        lambda rotations: rotations.unsqueeze(0).expand(
            5, *rotations.shape
        ).flatten(0, 1),

        # add noise to label
        lambda rotations: (
            label_pipe.noise(initial_point = rotations, time = time, num_samples = num_samples)
        ),


    ]
)

