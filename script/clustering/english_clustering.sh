#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 1-00:00:00

embeddings_folder="$1/LSC/SemEval-English"
labels_folder="$2/LSC/SemEval-English"
dataset_folder="$3/LSC/SemEval-English"
layers=12

declare -a algorithms=("app" "ap")
declare -a models=("bert-base-multilingual-cased" "bert-base-uncased")

for model in "${models[@]}"
do
	for algo in "${algorithms[@]}"
	do
		python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folder}/${model}" -t "${dataset_folder}/targets.txt"
	done
done
