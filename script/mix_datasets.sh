
model=bert-base-multilingual-cased
mix_dataset=MixDataset

declare -a all_dataset_folders=("$1/LSC/SemEval-English" "$1/LSC/SemEval-Latin" "$1/LSC/SemEval-German" "$1/LSC/SemEval-Swedish" "$1/LSC/LSCDiscovery-Spanish" "$1/LSC/DiacrIta-Italian" "$1/LSC/GlossReader12-Russian")
declare -a all_embeddings_folders=("$2/LSC/SemEval-English" "$2/LSC/SemEval-Latin" "$2/LSC/SemEval-German" "$2/LSC/SemEval-Swedish" "$2/LSC/LSCDiscovery-Spanish" "$2/LSC/DiacrIta-Italian" "$2/LSC/GlossReader12-Russian")
declare -a all_tokenization_folders=("$3/LSC/SemEval-English" "$3/LSC/SemEval-Latin" "$3/LSC/SemEval-German" "$3/LSC/SemEval-Swedish" "$3/LSC/LSCDiscovery-Spanish" "$3/LSC/DiacrIta-Italian" "$3/LSC/GlossReader12-Russian")
declare -a all_attentions_folders=("$4/LSC/SemEval-English" "$4/LSC/SemEval-Latin" "$4/LSC/SemEval-German" "$4/LSC/SemEval-Swedish" "$4/LSC/LSCDiscovery-Spanish" "$4/LSC/DiacrIta-Italian" "$4/LSC/GlossReader12-Russian")
declare -a all_scores_folders=("$5/LSC/SemEval-English" "$5/LSC/SemEval-Latin" "$5/LSC/SemEval-German" "$5/LSC/SemEval-Swedish" "$5/LSC/LSCDiscovery-Spanish" "$5/LSC/DiacrIta-Italian" "$5/LSC/GlossReader12-Russian")
declare -a all_labels_folders=("$6/LSC/SemEval-English" "$6/LSC/SemEval-Latin" "$6/LSC/SemEval-German" "$6/LSC/SemEval-Swedish" "$6/LSC/LSCDiscovery-Spanish" "$6/LSC/DiacrIta-Italian" "$6/LSC/GlossReader12-Russian")

declare -a all_targets="$1/LSC/SemEval-English/targets.txt" "$1/LSC/SemEval-Latin/targets.txt" "$1/LSC/SemEval-German/targets.txt" "$1/LSC/SemEval-Swedish/targets.txt" "$1/LSC/LSCDiscovery-Spanish/targets.txt" "$1/LSC/DiacrIta-Italian/targets.txt" "$1/LSC/GlossReader12-Russian/targets.txt")
declare -a all_binary="$1/LSC/SemEval-English/truth/binary.txt" "$1/LSC/SemEval-Latin/truth/binary.txt" "$1/LSC/SemEval-German/truth/binary.txt" "$1/LSC/SemEval-Swedish/truth/binary.txt" "$1/LSC/LSCDiscovery-Spanish/truth/binary.txt" "$1/LSC/DiacrIta-Italian/truth/binary.txt" "$1/LSC/GlossReader12-Russian/truth/binary.txt")
declare -a all_graded="$1/LSC/SemEval-English/truth/graded.txt" "$1/LSC/SemEval-Latin/truth/graded.txt" "$1/LSC/SemEval-German/truth/graded.txt" "$1/LSC/SemEval-Swedish/truth/graded.txt" "$1/LSC/LSCDiscovery-Spanish/truth/graded.txt" "$1/LSC/DiacrIta-Italian/truth/graded.txt" "$1/LSC/GlossReader12-Russian/truth/graded.txt")

mkdir -p "$1/LSC/${mix_dataset}"
mkdir -p "$1/LSC/${mix_dataset}/corpus1/token"
mkdir -p "$1/LSC/${mix_dataset}/corpus2/token"
mkdir -p "$1/LSC/${mix_dataset}/truth"
mkdir -p "$2/LSC/${mix_dataset}/${model}"
mkdir -p "$4/LSC/${mix_dataset}/${model}"
mkdir -p "$5/LSC/${mix_dataset}/${model}"
mkdir -p "$6/LSC/${mix_dataset}/${model}"

n=0
for dataset in "${all_dataset_folders[@]}"
do
   cat "${dataset}/corpus1/token/corpus1.txt" >> "$1/LSC/${mix_dataset}/corpus1/token/corpus1.txt" 
   cat "${dataset}/corpus2/token/corpus2.txt" >> "$1/LSC/${mix_dataset}/corpus2/token/corpus2.txt" 
   cat "${all_targets[n]}" >> "$1/LSC/${mix_dataset}/targets.txt" 
   cat "${all_binary[n]}" >> "$1/LSC/${mix_dataset}/binary.txt" 
   cat "${all_graded[n]}" >> "$1/LSC/${mix_dataset}/graded.txt" 

   cp "${all_embeddings_folders[n]}/${model}" -r "$2/LSC/${mix_dataset}/${model}" 
   cp "${all_tokenization_folders[n]}" -r "$3/LSC/${mix_dataset}" 
   cp "${all_attentions_folders[n]}/${model}" -r "$4/LSC/${mix_dataset}/${model}"
   cp "${all_labels_folders[n]}/${model}" -r "$6/LSC/${mix_dataset}/${model}"

   for layer in {1..12}
   do  
      cp "${all_scores_folders[n]}/${model}/${layer}/token.txt" >> "$5/LSC/${mix_dataset}/${model}/${layer}/token.txt"
   done
   n=$(($n+1))
done