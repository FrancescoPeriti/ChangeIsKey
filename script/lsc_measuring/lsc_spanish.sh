#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00

embedding_folder="$1/LSC/LSCDiscovery-Spanish"
label_folder="$2/LSC/LSCDiscovery-Spanish"
score_folder="$3/LSC/LSCDiscovery-Spanish"
dataset_folder="$4/LSC/LSCDiscovery-Spanish"

declare -a models=("bert-base-multilingual-cased" "dccuchile_bert-base-spanish-wwm-cased")

for model in "${models[@]}"
do
   python src/lsc_measuring.py -e "${embedding_folder}" -m "${model}" -L "${label_folder}" -o "${score_folder}/${model}" -t "${dataset_folder}/targets.txt"
done