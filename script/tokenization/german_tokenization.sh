dataset="$1/LSC/SemEval-German"
model="de_core_news_sm"
sampling=0
output="$2/LSC/SemEval-German"
tokenization_class="StandardSpacyTokenization"

python3 -m spacy download "${model}"
python3 "src/tokenization.py" -d "${dataset}" -m "${model}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"