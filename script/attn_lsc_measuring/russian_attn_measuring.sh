#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00


layers=12


declare -a models=("bert-base-multilingual-cased" "DeepPavlov/rubert-base-cased")
declare -a attentions=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a tokenized_datasets=("$3/LSC/GlossReader12-Russian" "$3/LSC/GlossReader23-Russian" "$3/LSC/GlossReader13-Russian")
declare -a outputs=("$4/LSC/GlossReader12-Russian" "$4/LSC/GlossReader23-Russian" "$4/LSC/GlossReader13-Russian")
declare -a targets=("$2/LSC/GlossReader12-Russian/targets_test.txt" "$2/LSC/GlossReader23-Russian/targets_test.txt" "$2/LSC/GlossReader13-Russian/targets_test.txt")


i=0
for tokenized_dataset in "${tokenized_datasets[@]}"
do
   for model in "${models[@]}"
   do
       python src/attn_lsc_measuring.py -a "${attentions[i]}" -m "${model}" -o "${outputs[i]}" -T "${tokenized_dataset}" -t "${targets[i]}" -l  "${layers}"
   done
    i=$(($i+1))
done
