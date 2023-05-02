#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU #N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00

declare -a models=("bert-base-multilingual-cased" "NbAiLab_nb-bert-base")
declare -a embeddings_folder=("$1/LSC/NorDiaChange12-Norwegian" "$1/LSC/NorDiaChange23-Norwegian")
declare -a labels_folder=("$2/LSC/NorDiaChange12-Norwegian" "$2/LSC/NorDiaChange23-Norwegian")
declare -a scores_folder=("$3/LSC/NorDiaChange12-Norwegian" "$3/LSC/NorDiaChange23-Norwegian")
declare -a targets_filename=("$4/LSC/NorDiaChange12-Norwegian/targets.txt" "$4/LSC/NorDiaChange23-Norwegian/targets.txt")


i=0
for embedding_folder in "${embeddings_folder[@]}"
do
   for model in "${models[@]}"
   do
       python src/lsc_measuring.py -e "${embedding_folder}" -m "${model}" -L "${labels_folder[i]}" -o "${scores_folder[i]}/${model}" -t "${targets_filename[i]}"
   done
    i=$(($i+1))
done


