"""
This script provides functionality to unpack AMASS datasets, extract human poses
in the axis-angle representation and store them in a single tensor in a pt file.

Author          : Rezha Adrian Tanuharja
Mail            : tanuharja@ias.uni-stuttgart.de
Date created    : 2024.08.14

Usage (executed from project root dir):

    1. Modify the following variables accordingly:
        - `compressedFilesDirectory`: where you store .tar.bz2 files
        - `extractedFilesDirectory`: where to put unpacked files
        - `saveFile` : a file with extension `.pt` to store the tensor

    2. execute from project root dir `python utils/preprocessAMASS`

"""


import subprocess
from preprocessDatasets import store_datasets_as_tensors

def main() -> None:

    # -- Specify directories
    compressed_files_directory = "./downloads"
    extracted_files_directory = "./extractedData"
    saveFile = "./preprocessedData/axisAngleTensors.pt"


    # -- Extract compressed datasets downloaded from AMASS
    try:

        _ = subprocess.run(
            [
                "./utils/preprocessAMASS/extractCompressedDatasets.sh",
                "-i", compressed_files_directory,
                "-o", extracted_files_directory
            ],
            capture_output=True,    # do not output to terminal, store in stdout and stderr
            text=True,              # ensure output is captured as text
            check=True              # raise exception if fails to run
        )

    except FileNotFoundError:
        print("The script './utils/preprocessAMASS/extractCompressedDatasets.sh' is not found")
        return

    except subprocess.CalledProcessError as e:
        print(f"Error while extracting datasets: {e.stderr}")
        return

    except Exception as e:
        print(f"Unexpected error occured: {str(e)}")
        return

    # -- Convert all extracted datasets into a (PyTorch)tensors and save in a .pt file
    store_datasets_as_tensors(
        inputDirectory = compressed_files_directory,
        outputFile = saveFile
    )

    # -- Remove extracted datasets from extractedFilesDirectory
    _ = subprocess.run(f"rm -r {extracted_files_directory}/*", shell=True)


if __name__ == "__main__":
    main()
