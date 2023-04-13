#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
tokenization_folder="$2/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
tokenization_folder="$(realpath ${tokenization_folder})"
norwegian=NorDiaChange-Norwegian

echo "${current_folder}"
git clone https://github.com/ltgoslo/nor_dia_change.git
python "${current_folder}/src/processing/norwegian_processing.py" -d "${dataset_folder}/${norwegian}" -t "${tokenization_folder}/${norwegian}"
cut -f1 nor_dia_change/subset1/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange12-Norwegian/targets.txt"
cut -f1 nor_dia_change/subset2/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange23-Norwegian/targets.txt"
mkdir "${dataset_folder}/NorDiaChange12-Norwegian/truth/"
mkdir "${dataset_folder}/NorDiaChange23-Norwegian/truth/"
cut -f1,12 nor_dia_change/subset1/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange12-Norwegian/truth/binary.txt"
cut -f1,15 nor_dia_change/subset1/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange12-Norwegian/truth/graded.txt"
cut -f1,12 nor_dia_change/subset2/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange23-Norwegian/truth/binary.txt"
cut -f1,15 nor_dia_change/subset2/stats/stats_groupings.tsv  | tail -n+2 > "${dataset_folder}/NorDiaChange23-Norwegian/truth/graded.txt"
rm -rf nor_dia_change
