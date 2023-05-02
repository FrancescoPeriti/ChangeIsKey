#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 1-00:00:00

layers=12

declare -a embeddings_folders=("$1/LSC/NorDiaChange12-Norwegian" "$1/LSC/NorDiaChange23-Norwegian")
declare -a labels_folders=("$2/LSC/NorDiaChange12-Norwegian" "$2/LSC/NorDiaChange23-Norwegian")
declare -a dataset_folders=("$3/LSC/NorDiaChange12-Norwegian" "$3/LSC/NorDiaChange23-Norwegian")
declare -a algorithms=("app" "ap")
declare -a models=("bert-base-multilingual-cased" "NbAiLab_nb-bert-base")

for model in "${models[@]}"
do
    n=0
    for embeddings_folder in "${embeddings_folders[@]}"
    do
	for algo in "${algorithms[@]}"
	do
	    python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folders[n]}/${model}" -t "${dataset_folders[n]}/targets.txt"
	done
     n=$(($n+1))
    done
done
