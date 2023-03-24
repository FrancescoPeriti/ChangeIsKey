#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 1-00:00:00

labels_folder="$2/LSC/GlossReader12-Russian"
dataset_folder="$3/LSC/GlossReader12-Russian"
layers=12

declare -a embeddings_folders=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a algorithms=("app" "ap")
declare -a models=("bert-base-multilingual-cased" "bert-base-uncased")

for model in "${models[@]}"
do
	for embeddings_folder in "${embeddings_folders[@]}"
	do
		for algo in "${algorithms[@]}"
		do
			python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folder}/${model}" -t "${dataset_folder}/targets.txt"
		done
	done
done
