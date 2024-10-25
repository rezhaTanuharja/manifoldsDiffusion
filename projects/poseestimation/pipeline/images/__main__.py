"""
poseestimation.pipeline.images.__main__
=======================================

Mainly used for debugging the image pipeline
"""


import torch
import resnet
import numpy


def main():

    device = torch.device('cpu')
    images = numpy.random.random(size = (20, 244, 244, 3))

    try:
        image_pipeline = resnet.create_image_pipeline(
            num_sample_duplicates = 5,
            num_timestamps = 3,
            device = device
        )

    except Exception as e:
        print(f"Failed to create an image pipeline: {type(e)}")
        raise

    results = image_pipeline(images)

    print(results)


if __name__ == "__main__":

    try:
        main()

    except Exception as e:
        print(f"Failed to run main: {type(e)}")
