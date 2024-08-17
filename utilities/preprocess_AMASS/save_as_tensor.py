"""
This script contains a function to preprocess AMASS datasets after they have
been unpacked from the .tar.bz2 files.

Author          : Rezha Adrian Tanuharja
Mail            : tanuharja@ias.uni-stuttgart.de
Date created    : 2024.08.14

Usage:

    executed by utilities/preprocess_AMASS/__main__.py

"""


import os
import numpy as np
import torch


def store_datasets_as_tensors(
    input_directory: str,
    output_file: str
) -> None:
    """
    Extract axis angle (PyTorch)tensors from .npz files, stack them, and store
    it in a single tensor in a file with extension `.pt`.

    Parameters:
    input_directory (str): the location of .npz files
    output_file (str)    : the file to save the stacked tensors (.pt extension)
    """

    # -- Count the number of samples (total from all '.npz' files)
    sample_count = 0

    for path, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.npz'):

                try:
                    data = np.load(os.path.join(path, file))
                    sample_count += data['poses'].shape[0]

                except:
                    continue


    # -- Pre-allocate memory for tensors
    total_num_dimensions = 156 + 1
    all_tensors  = torch.zeros(sample_count, total_num_dimensions)


    # -- Extract tensors from each file and store in allTensors
    sample_index = 0

    for path, _, files in os.walk(input_directory):

        for file in files:

            if file.endswith('.npz') and not file.startswith('.') and ("neutral_stage" not in file):

                try:
                    data = np.load(os.path.join(path, file))['poses']

                except:
                    continue

                for idx in range(data.shape[0]):

                    try:
                        axis_angle = np.append(data[idx], 0)
                        all_tensors[sample_index] = torch.tensor(axis_angle)

                    except:
                        continue

                    sample_index += 1

    all_tensors = all_tensors[:sample_index]

    # -- Save allTensors to outputFile
    torch.save(all_tensors, output_file)
