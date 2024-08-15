"""
@file utils/preprocessAMASS/storeDatasetsAsTensors.py

@brief
Contains a function to preprocess extracted AMASS dataset

@author Rezha Adrian Tanuharja
@date 2024.08.14

Intended usage:

    executed by utils/preprocessAMASS/__main__.py
"""


import os
import numpy as np
import torch


def storeDatasetsAsTensors(
    inputDirectory: str,
    outputFile: str
) -> None:
    """
    @brief
    Extract axis angle (PyTorch)tensors from .npz files, stack them, and store it in a .pt file.

    @param inputDirectory   the location of .npz files (files may be in its subdirectories)
    @param outputFile       the file to save the stacked tensors
    """


    # -- Count the number of samples (the number of '.npz' files)

    sampleCount = 0

    for path, _, files in os.walk(inputDirectory):
        for file in files:
            if file.endswith('.npz'):

                try:
                    data = np.load(os.path.join(path, file))
                    sampleCount += data['poses'].shape[0]

                except:
                    continue


    # -- Pre-allocate memory for tensors

    tensorShape = (52, 3)
    allTensors  = torch.zeros(sampleCount, *tensorShape)


    # -- Extract tensors from each file and store in allTensors

    sampleIndex = 0

    for path, _, files in os.walk(inputDirectory):

        for file in files:

            if file.endswith('.npz') and not file.startswith('.') and ("neutral_stage" not in file):

                try:
                    data = np.load(os.path.join(path, file))['poses']

                except:
                    continue

                for idx in range(data.shape[0]):

                    try:
                        axisAngle = data[idx].reshape(52, 3)
                        allTensors[sampleIndex] = torch.tensor(axisAngle)

                    except:
                        continue

                    sampleIndex += 1

    allTensors = allTensors[:sampleIndex]


    # -- Save allTensors to outputFile

    torch.save(allTensors, outputFile)
