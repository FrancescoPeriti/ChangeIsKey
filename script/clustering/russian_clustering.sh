#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 1-00:00:00

layers=12

declare -a embeddings_folders=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a labels_folders=("$2/LSC/GlossReader12-Russian" "$2/LSC/GlossReader23-Russian" "$2/LSC/GlossReader13-Russian")
declare -a dataset_folders=("$3/LSC/GlossReader12-Russian" "$3/LSC/GlossReader23-Russian" "$3/LSC/GlossReader13-Russian")
declare -a algorithms=("app" "ap")
declare -a models=("bert-base-multilingual-cased" "DeepPavlov_rubert-base-cased")

for model in "${models[@]}"
do
    n=0
    for embeddings_folder in "${embeddings_folders[@]}"
    do
	for algo in "${algorithms[@]}"
	do
	    python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folders[n]}/${model}" -t "${dataset_folders[n]}/targets_test.txt"
	done
     n=$(($n+1))
    done
done
