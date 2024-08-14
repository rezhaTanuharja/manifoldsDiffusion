"""
@file utils/preprocessAMASS/__main__.py

@brief
A script to preprocess AMASS dataset

@author Rezha Adrian Tanuharja
@date 2024.08.14

Usage (from project root dir): python utils/preprocessAMASS
"""


import subprocess
from storeDatasetsAsTensors import *

def main():


    # -- Extract compressed datasets downloaded from AMASS

    try:

        _ = subprocess.run(
            ["./utils/preprocessAMASS/extractCompressedDatasets.sh"],
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


    # -- Convert extracted datasets into PyTorch tensors and save as .pt files

    #TODO


if __name__ == "__main__":
    main()
