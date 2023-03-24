#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 5:00:00

tokenized_dataset="$1/LSC/LSCDiscovery-Spanish"
max_length=512
layers=12
batch_size=64
output="$2/LSC/LSCDiscovery-Spanish"
sampling=250
targets="$3/LSC/LSCDiscovery-Spanish/targets.txt"
agg_sub_words='mean'

declare -a models=("bert-base-multilingual-cased" "dccuchile/bert-base-spanish-wwm-cased")

for model in "${models[@]}"
do
   python src/embeddings.py -t "${tokenized_dataset}" -m "${model}" -M "${max_length}" -l "${layers}" -b "${batch_size}" -o "${output}" -n "${sampling}" -T "${targets}"
done
