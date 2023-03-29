#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 2-00:00:00

main_folder="$1"
dataset="SemEval-Latin"
layers=12
output_folder="$2"

declare -a models=("bert-base-multilingual-cased")

for model in "${models[@]}"
do
   python src/brute_force.py -f "${main_folder}" -d "${dataset}" -m "${model}" -l "${layers}" -D 3 -o "${output_folder}"
done