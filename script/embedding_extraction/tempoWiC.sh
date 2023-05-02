#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 1:00:00

tokenized_dataset="$1/WiC/TempoWiC"
output="$2/WiC/TempoWiC"


declare -a models=("bert-base-multilingual-cased" "bert-base-uncased" "cambridgeltl/mirrorwic-bert-base-uncased")

for model in "${models[@]}"
do
    python src/embeddings_wic.py -t "${tokenized_dataset}/train" -m "${model}" -o "${output}/train"
    python src/embeddings_wic.py -t "${tokenized_dataset}/validation" -m "${model}" -o "${output}/test"
done
