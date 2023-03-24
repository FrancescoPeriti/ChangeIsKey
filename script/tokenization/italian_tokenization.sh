dataset="$1/LSC/SemEval-Italian"
model="it_core_news_sm"
sampling=0
output="$2/LSC/SemEval-Italian"
tokenization_class="ItalianSpacyTokenization"

python3 -m spacy download "${model}"
python3 "src/tokenization.py" -d "${dataset}" -m "${model}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"