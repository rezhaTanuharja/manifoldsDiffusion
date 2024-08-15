#!/bin/bash

##
# @file utils/preprocessAMASS/extractCompressedDatasets.sh
#
# @brief
# This script extracts .tar.bz2 files from inputDirectory and store in outputDirectory
#
# @author Rezha Adrian Tanuharja
# @date 2024.08.14
#
# Intended usage:
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

inputDirectory="./downloads"
outputDirectory="./extractedData"


# -- Parse flags (-i or --input) and (-o or --output)

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -i|--input) inputDirectory="$2"; shift ;;
    -o|--output) outputDirectory="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done


# -- Extract all files with .tar.bz2 extension

for file in "$inputDirectory"/*.tar.bz2; do

  tar -xjf "$file" -C "$outputDirectory"
done
