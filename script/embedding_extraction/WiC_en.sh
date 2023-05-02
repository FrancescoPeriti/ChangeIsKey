#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 1:00:00

tokenized_dataset="$1/WiC/WiC-English"
output="$2/WiC/WiC-English"


declare -a models=("bert-base-multilingual-cased") #, "cambridgeltl/mirrorwic-bert-base-uncased") #("bert-base-uncased") #"bert-base-multilingual-cased" 

for model in "${models[@]}"
do
    python src/embeddings_wic.py -t "${tokenized_dataset}/dev" -m "${model}" -o "${output}/dev"
    python src/embeddings_wic.py -t "${tokenized_dataset}/train" -m "${model}" -o "${output}/train"
    python src/embeddings_wic.py -t "${tokenized_dataset}/test" -m "${model}" -o "${output}/test"
done
