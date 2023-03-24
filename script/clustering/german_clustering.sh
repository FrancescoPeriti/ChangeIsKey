#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 1-00:00:00

embeddings_folder="$1/LSC/SemEval-German"
labels_folder="$2/LSC/SemEval-German"
dataset_folder="$3/LSC/SemEval-German"
layers = 12

declare -a algorithms=("app" "ap")
declare -a models=("bert-base-multilingual-cased" "bert-base-german-cased")

for model in "${models[@]}"
do
	for algo in "${algorithms[@]}"
	do
		python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folder}/${model}"-t "${dataset_folder}/targets.txt"
	done
done