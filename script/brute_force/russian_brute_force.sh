#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 6-00:00:00

main_folder="$1"
layers=12
output_folder="$2"

declare -a models=("bert-base-multilingual-cased" "DeepPavlov_rubert-base-cased")
declare -a datasets=("GlossReader12-Russian" "GlossReader23-Russian" "GlossReader13-Russian")

i=0
for dataset in "${datasets[@]}"
do
   for model in "${models[@]}"
   do
       python src/brute_force.py -f "${main_folder}" -d "${dataset}" -m "${model}" -l "${layers}" -D 3 -o "${output_folder}"
   done
    i=$(($i+1))
done
