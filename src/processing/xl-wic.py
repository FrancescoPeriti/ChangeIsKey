import pandas as pd
import csv
from tqdm import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(prog='Processing XL-WiC')
parser.add_argument('-d', '--dataset_path', type=str, help='Path to the corpus to process')
parser.add_argument('-t', '--tokenization_path', type=str, help='Path to the processed corpus')
args = parser.parse_args()

dataset_path=args.dataset_path
tokenization_path=args.tokenization_path

for lang_dir in ['WiC-XL/xlwic_wikt/german_de', 'WiC-XL/xlwic_wikt/italian_de']:
    dir_path = f'{dataset_path}/{lang_dir}'

    iso = lang_dir.split('/')[-1].split('_')[1]

    if iso == 'de':
        lang = 'German'
    elif iso == 'it':
        lang = 'Italan'
    elif iso == 'fr':
        lang = 'French'


    valid_path = f'{dir_path}/{iso}_valid.txt'

    columns = ['lemma', 'pos', 'start1', 'end1', 'start2', 'end2', 'sent1', 'sent2', 'gold']
    columns_test = columns[:-1]
    
    for k in ['valid', 'train', 'test']:
        k_set = k if k!='valid' else 'dev'
        Path(f'{dataset_path}/WiC-{lang}/{k_set}').mkdir(parents=True, exist_ok=True)
        output_data = f'{dataset_path}/WiC-{lang}/{k_set}/{k_set}.data.txt'
        output_gold = f'{dataset_path}/WiC-{lang}/{k_set}/{k_set}.gold.txt'

        if k != 'test':
            sample_path = f'{dir_path}/{iso}_{k}.txt'
            df = pd.read_csv(sample_path, sep='\t', quoting=csv.QUOTE_NONE, names=columns)
            gold = [f'{i}\n' for i in df.gold.values]
        else:
            sample_path = f'{dir_path}/{iso}_{k}_data.txt'
            df = pd.read_csv(sample_path, sep='\t', quoting=csv.QUOTE_NONE, names=columns_test)
            gold_path = f'{dir_path}/{iso}_{k}_gold.txt'
            gold = pd.read_csv(gold_path, sep='\t', quoting=csv.QUOTE_NONE, names=["gold"])
            gold = [f'{i}\n' for i in gold.gold.values]

        df.to_csv(output_data, index=False, sep='\t')
        open(output_gold, mode='w').writelines(gold)

        tokenize(df)

        if k == 'test':
            for s in ['IV', 'OOV']:
                sample_path = f'{dir_path}/{s}/{iso}_{s.lower()}_test_data.txt'
                gold_path = f'{dir_path}/{s}/{iso}_{s.lower()}_test_gold.txt'
                df = pd.read_csv(sample_path, sep='\t', quoting=csv.QUOTE_NONE, names=columns_test)
                gold = pd.read_csv(gold_path, sep='\t', quoting=csv.QUOTE_NONE, names=["gold"])
                gold = [f'{i}\n' for i in gold.gold.values]

                df.to_csv(output_data.replace(f'{k_set}.data', f'{k_set}_{s.lower()}.data', index=False, sep='\t')
                open(output_gold.replace(f'{k_set}.gold', f'{k_set}_{s.lower()}.gold', mode='w').writelines(gold)
                tokenize(df)


                     gold = pd.read_csv(f'TempoWiC/data/{folder}.labels.tsv', sep='\t', names=['id', 'gold'])
    data = pd.read_json(f'TempoWiC/data/{folder}.data.jl', lines=True)
    data['sent1'] = data['tweet1'].apply(lambda x: x['text'])
    data['sent2'] = data['tweet2'].apply(lambda x: x['text'])
    
    folder_out=folder if folder=="train" else "test"
    Path(f'{dataset_path}/{folder_out}').mkdir(parents=True, exist_ok=True)
    gold['gold'].to_csv(f'{dataset_path}/{folder_out}/{folder_out}.gold.txt', sep='\t', index=False, header=None)
    data.to_csv(f'{dataset_path}/{folder_out}/{folder_out}.data.txt', sep='\t', index=False, header=None)
    
    df = pd.read_json(f'TempoWiC/data/{folder}.data.jl', lines=True)
    
    results = list()

    c = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        word = row['word']
        for sent in ['tweet1', 'tweet2']:
            start, end = row[sent]['text_start'], row[sent]['text_end']
            token = row[sent]['text'][start:end]
            
            sent=row[sent]['text']
            sent=sent[:start]+" "+sent[start:end]+' '+sent[end:]
            sent=" ".join(sent.split())
            start=sent.find(token)
            end=start+len(token)
            
            results.append(dict(sentidx=c, lemma=row['word'], token=sent[start:end], start=start, end=end, sent=sent.strip()))
    
    Path(tokenization_path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_json(filename, orient='records', lines=True)
