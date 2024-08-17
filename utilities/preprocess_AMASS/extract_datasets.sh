#!/bin/bash

##
# This script extracts AMASS datasets from .tar.bz2 files in inputDirectory and
# store them inside outputDirectory.
#
# Author          : Rezha Adrian Tanuharja
# Mail            : tanuharja@ias.uni-stuttgart.de
# Date created    : 2024.08.14
#
# Usage:
#
#   executed by utils/preprocessAMASS/__main__.py
#
#   or
#
#   (from project root dir): source ./utils/extractCompressedDatasets.sh
#   this extract files in './downloads' and store in './extractedData'
#
#   optionally the input and output directory can be specified with -i and -o flag, e.g.,
#   source ./utils/extractCompressedDatasets.sh -i './input' -o './output'
#


# -- These are the default input and output directories
input_directory="./downloads"
output_directory="./extractedData"

# -- Parse flags (-i or --input) and (-o or --output)
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -i|--input) input_directory="$2"; shift ;;
    -o|--output) output_directory="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# -- Extract all files with .tar.bz2 extension
for file in "$input_directory"/*.tar.bz2; do
  tar -xjf "$file" -C "$output_directory"
done
