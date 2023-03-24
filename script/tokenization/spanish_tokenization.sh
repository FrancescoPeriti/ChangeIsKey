dataset="$1/LSC/LSCDiscovery-Spanish"
sampling=0
output="$2/LSC/LSCDiscovery-Spanish"
tokenization_class="StandardTokenization"

python3 "src/tokenization.py" -d "${dataset}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"