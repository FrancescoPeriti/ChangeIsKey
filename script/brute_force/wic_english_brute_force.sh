#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 4-00:00:00

main_folder="$1"
dataset="SemEval-English"
layers=12
output_folder="$2"
stable_words_to_ignore="$3"


python src/brute_force_wic_en.py
