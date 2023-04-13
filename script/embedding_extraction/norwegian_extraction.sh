#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 5:00:00

declare -a models=("bert-base-multilingual-cased" "NbAiLab/nb-bert-base")
declare -a tokenized_datasets=("$1/LSC/NorDiaChange12-Norwegian" "$1/LSC/NorDiaChange23-Norwegian")
declare -a outputs=("$2/LSC/NorDiaChange12-Norwegian" "$2/LSC/NorDiaChange23-Norwegian")
declare -a targets=("$3/LSC/NorDiaChange12-Norwegian/targets.txt" "$3/LSC/NorDiaChange23-Norwegian/targets.txt")

max_length=512
layers=12
batch_size=64
sampling=0
agg_sub_words='mean'

i=0
for tokenized_dataset in "${tokenized_datasets[@]}"
do
   for model in "${models[@]}"
   do
      python src/embeddings.py -t "${tokenized_dataset}" -m "${model}" -M "${max_length}" -l "${layers}" -b "${batch_size}" -o "${outputs[i]}" -n "${sampling}" -T "${targets[i]}"
   done
    i=$(($i+1))
done
