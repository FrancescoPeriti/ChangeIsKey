#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00

attention_folder="$1/LSC/SemEval-Latin"
dataset_folder="$2/LSC/SemEval-Latin"
tokenization_folder="$3/LSC/SemEval-Latin"
attn_score_folder="$4/LSC/SemEval-Latin"

layers=12

declare -a models=("bert-base-multilingual-cased")

for model in "${models[@]}"
do
    python src/attn_lsc_measuring.py -a "${attention_folder}" -m "${model}" -o "${attn_score_folder}" -T "${tokenization_folder}" -t "${dataset_folder}/targets.txt" -l  "${layers}"
done
