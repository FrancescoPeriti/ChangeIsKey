main_folder_data="/mimer/NOBACKUP/groups/cik_data/"
datasets_folder="${main_folder_data}/datasets"
tokenization_folder="${main_folder_data}/tokenization"
embeddings_folder="${main_folder_data}/contextualized_embeddings"
attentions_folder="${main_folder_data}/attentions"
ft_models_folder="${main_folder_data}/ft_models"
labels_folder="${main_folder_data}/labels"
score_folder="${main_folder_data}/scores"
attn_score_folder="${main_folder_data}/attn_scores"
brute_force_folder="${main_folder_data}/brute_force"

# Absolute paths
datasets_folder="$(realpath ${datasets_folder})"
tokenization_folder="$(realpath ${tokenization_folder})"
embeddings_folder="$(realpath ${embeddings_folder})"
attentions_folder="$(realpath ${attentions_folder})"
ft_models_folder="$(realpath ${ft_models_folder})"
labels_folder="$(realpath ${labels_folder})"
score_folder="$(realpath ${score_folder})"
attn_score_folder="$(realpath ${attn_score_folder})"

#module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
source venv/bin/activate

# -- Download --
#echo "Russian data available here: https://disk.yandex.ru/d/CIU9Hm0tvKPH2g"
#echo "Download it and rename it data.zip"
#bash "script/download/english_download.sh" "${datasets_folder}"
#bash "script/download/russian_download.sh" "${datasets_folder}" "${tokenization_folder}"
#bash "script/download/italian_download.sh" "${datasets_folder}"
#bash "script/download/german_download.sh" "${datasets_folder}"
#bash "script/download/swedish_download.sh" "${datasets_folder}"
#bash "script/download/latin_download.sh" "${datasets_folder}"
#bash "script/download/spanish_download.sh" "${datasets_folder}"
bash "script/download/norwegian_download.sh" "${datasets_folder}" "${tokenization_folder}"

# -- Tokenization --
#sbatch "script/tokenization/english_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
#sbatch "script/tokenization/german_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
#bash "script/tokenization/italian_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
#bash "script/tokenization/latin_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
#sbatch "script/tokenization/swedish_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
#sbatch "script/tokenization/spanish_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"

# -- Embeddings extraction --
#sbatch "script/embedding_extraction/english_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/german_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/italian_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/latin_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/swedish_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/spanish_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
#sbatch "script/embedding_extraction/russian_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
sbatch "script/embedding_extraction/norwegian_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"

# -- Attentions extraction --
##sbatch "script/attention_extraction/english_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
##sbatch "script/attention_extraction/german_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
##sbatch "script/attention_extraction/italian_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
##sbatch "script/attention_extraction/latin_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
#sbatch "script/attention_extraction/swedish_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
##sbatch "script/attention_extraction/spanish_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
#sbatch "script/attention_extraction/russian_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"

# -- Finetuning --
#bash "script/bert_finetuning/english_finetuning.sh" "${datasets_folder}" "${tokenization_folder}" "${ft_models_folder}"

# -- Clustering --
#sbatch "script/clustering/english_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#bash "script/clustering/italian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#sbatch "script/clustering/russian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#sbatch "script/clustering/german_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#sbatch "script/clustering/latin_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#sbatch "script/clustering/swedish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
#sbatch "script/clustering/spanish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"

# - LSC measuring --
##sbatch "script/lsc_measuring/english_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
##bash "script/lsc_measuring/italian_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
#bash "script/lsc_measuring/russian_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
##bash "script/lsc_measuring/german_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
#sbatch "script/lsc_measuring/latin_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
##bash "script/lsc_measuring/swedish_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"
##bash "script/lsc_measuring/spanish_measuring.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}" "${datasets_folder}"

# -- Mix datasets --
#bash "script/mix_datasets.sh" "${datasets_folder}" "${embeddings_folder}" "${tokenization_folder}" "${attentions_folder}" "${score_folder}" "${labels_folder}"

# -- Brute force --
#sbatch "script/brute_force/english_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/spanish_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/german_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/latin_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/swedish_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/russian_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 0
#sbatch "script/brute_force/english_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4
#sbatch "script/brute_force/spanish_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4
#sbatch "script/brute_force/german_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4
#sbatch "script/brute_force/latin_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4
#sbatch "script/brute_force/swedish_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4
#sbatch "script/brute_force/russian_brute_force.sh" "${main_folder_data}" "${brute_force_folder}" 4


# -- Attn scores --
#sbatch "script/attn_lsc_measuring/english_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/latin_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/spanish_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/swedish_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/german_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/russian_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"
#sbatch "script/attn_lsc_measuring/italian_attn_measuring.sh" "${attentions_folder}" "${datasets_folder}" "${tokenization_folder}" "${attn_score_folder}"