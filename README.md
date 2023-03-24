datasets_folder="datasets"
tokenization_folder="tokenization"
embeddings_folder="contextualized_embeddings"
attentions_folder="attentions"
ft_models_folder="ft_models"
labels_folder="labels"
score_folder="scores"

# Absolute paths
datasets_folder="$(realpath ${datasets_folder})"
tokenization_folder="$(realpath ${tokenization_folder})"
embeddings_folder="$(realpath ${embeddings_folder})"
attentions_folder="$(realpath ${attentions_folder})"
ft_models_folder="$(realpath ${ft_models_folder})"
labels_folder="$(realpath ${labels_folder})"
score_folder="$(realpath ${score_folder})"

module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# -- Download --
echo "Russian data available here: https://disk.yandex.ru/d/CIU9Hm0tvKPH2g"
echo "Download it and rename it data.zip"
bash "script/download/english_download.sh" "${datasets_folder}"
bash "script/download/russian_download.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/download/italian_download.sh" "${datasets_folder}"
bash "script/download/german_download.sh" "${datasets_folder}"
bash "script/download/swedish_download.sh" "${datasets_folder}"
bash "script/download/latin_download.sh" "${datasets_folder}"
bash "script/download/spanish_download.sh" "${datasets_folder}"

# -- Tokenization --
bash "script/tokenization/english_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/tokenization/german_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/tokenization/italian_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/tokenization/latin_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/tokenization/swedish_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"
bash "script/tokenization/spanish_tokenization.sh" "${datasets_folder}" "${tokenization_folder}"

# -- Embeddings extraction --
bash "script/embedding_extraction/english_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/german_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/italian_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/latin_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/swedish_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/spanish_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"
bash "script/embedding_extraction/russian_extraction.sh" "${tokenization_folder}" "${embeddings_folder}" "${datasets_folder}"

# -- Attentions extraction --
bash "script/attention_extraction/english_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/german_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/italian_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/latin_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/swedish_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/spanish_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"
bash "script/attention_extraction/russian_extraction.sh" "${tokenization_folder}" "${attentions_folder}" "${datasets_folder}"

# -- Finetuning --
bash "script/bert_finetuning/english_finetuning.sh" "${datasets_folder}" "${tokenization_folder}" "${ft_models_folder}"

# -- Clustering --
bash "script/clustering/english_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/italian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/russian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/german_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/latin_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/swedish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"
bash "script/clustering/spanish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${datasets_folder}"

# - LSC measuring --
bash "script/lsc_measuring/english_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/italian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/russian_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/german_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/latin_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/swedish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"
bash "script/lsc_measuring/spanish_clustering.sh" "${embeddings_folder}" "${labels_folder}" "${score_folder}"

# -- Mix datasets --
bash "script/mix_datasets.sh" "${datasets_folder}" "${embeddings_folder}" "${tokenization_folder}" "${attentions_folder}" "${score_folder}" "${labels_folder}"