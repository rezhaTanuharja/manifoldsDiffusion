"""
poseestimation.pipeline.images.resnet
=====================================

Generate image features using a pretrained ResNet50 model

Functions
---------
`create_image_pipeline` :
    create an image pipeline containing the Resnet model in a given device
"""


import diffusionmodels.dataprocessing as dataprocessing
import torch
import torchvision.models as models


def create_image_pipeline(
    num_sample_duplicates: int,
    num_timestamps: int,
    device: torch.device
) -> dataprocessing.Pipeline:

    try:
        residual_network = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        residual_network = residual_network.to(device)
        residual_network = residual_network.eval()

    except Exception as e:
        print(f"Failed to load ResNet50: {type(e)}")
        raise

    image_pipeline = dataprocessing.Pipeline(
        transforms = [

            # convert a NumPy array into a Torch tensor
            lambda images: torch.tensor(images, dtype = torch.float),

            # send tensor to the assigned computing device
            lambda images: images.to(device),

            # in TF sample shape is 244, 244, 3 while in Torch it should be 3, 244, 244
            lambda images: images.permute(0, 3, 1, 2),

            # use a pretrained resnet model to process images
            lambda images: residual_network(images),

            # duplicate images to noise each sample multiple times
            lambda images: images.unsqueeze(0).expand(
                num_sample_duplicates, *images.shape
            ).flatten(0, 1),

            # duplicate images to match the number of noise levels
            lambda images: images.unsqueeze(0).expand(
                num_timestamps, *images.shape
            ).flatten(0, 1),

        ]
    )

    return image_pipeline
