#!/bin/bash

##
# @file utils/preprocessAMASS/extractCompressedDatasets.sh
#
# @brief
# This script extracts dataset in 'downloads' and store it inside 'data'
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
#   (from project root dir): source ./utils/extractDatasets.sh
#

# -- specify the location of compressed data
inputDirectory="./downloads"

# -- specify the location to store decompressed data
outputDirectory="./data"

# -- loop through all files in inputDirectory with .tar.bz2 extension
for file in "$inputDirectory"/*.tar.bz2; do

  # -- extract each file and store in outputDirectory
  tar -xjf "$file" -C "$outputDirectory"

done
