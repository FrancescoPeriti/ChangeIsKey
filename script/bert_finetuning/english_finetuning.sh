#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 2-00:00:00

dataset_folder="$1/LSC/SemEval-English"
tokenization_folder="$2/LSC/SemEval-English"
model="bert-base-uncased"
max_length=512
n_epochs=5
batch_size=8

finetune_class='BertFineTuningFull'
model_output="$3/LSC/SemEval-English/bert-base-uncased-full"
python src/finetuning.py -t "${dataset_folder}" -m "${model}" -M "${max_length}" -e "${n_epochs}" -o "${model_output}" -b "${batch_size}" -f "${finetune_class}"

finetune_class='BertFineTuningFilter'
model_output="$3/LSC/SemEval-English/bert-base-uncased-filter"
python src/finetuning.py -t "${tokenization_folder}" -m "${model}" -M "${max_length}" -e "${n_epochs}" -o "${model_output}" -b "${batch_size}" -f "${finetune_class}"

finetune_class='BertFineTuningSeparate'
model_output="$3/LSC/SemEval-English/bert-base-uncased-separate"
python src/finetuning.py -t "${tokenization_folder}" -m "${model}" -M "${max_length}" -e "${n_epochs}" -o "${model_output}" -b "${batch_size}" -f "${finetune_class}" -T "${dataset_folder}/targets.txt"