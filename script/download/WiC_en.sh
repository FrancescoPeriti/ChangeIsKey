#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/WiC/"
tokenization_folder="$2/WiC/"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"

python src/processing/WiC_en.py -d "${dataset_folder}/WiC-English/" -t "${tokenization_folder}/WiC-English/"
