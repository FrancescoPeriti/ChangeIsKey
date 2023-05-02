#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/WiC/"
tokenization_folder="$2/WiC/"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"

git clone https://github.com/cardiffnlp/TempoWiC.git
python src/processing/tempoWiC.py -d "${dataset_folder}/TempoWiC/" -t "${tokenization_folder}/TempoWiC/"
rm -rf TempoWiC
