"""
@file utils/preprocessAMASS/__main__.py

@brief
A script to preprocess AMASS dataset

@author Rezha Adrian Tanuharja
@date 2024.08.14

Usage (from project root dir): python utils/preprocessAMASS
"""


import subprocess
from preprocessDatasets import storeDatasetsAsTensors

def main() -> None:


    # -- Specify directories
    
    compressedFilesDirectory = "./downloads"
    extractedFilesDirectory = "./extractedData"
    saveFile = "./preprocessedData/axisAngleTensors.pt"


    # -- Extract compressed datasets downloaded from AMASS

    try:

        _ = subprocess.run(
            [
                "./utils/preprocessAMASS/extractCompressedDatasets.sh",
                "-i", compressedFilesDirectory,
                "-o", extractedFilesDirectory
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

    storeDatasetsAsTensors(
        inputDirectory = compressedFilesDirectory,
        outputFile = saveFile
    )


    # -- Remove extracted datasets from extractedFilesDirectory

    _ = subprocess.run(f"rm -r {extractedFilesDirectory}/*", shell=True)


if __name__ == "__main__":
    main()
