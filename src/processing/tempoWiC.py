import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(prog='Processing TempoWiC')
parser.add_argument('-d', '--dataset_path', type=str, help='Path to the corpus to process')
parser.add_argument('-t', '--tokenization_path', type=str, help='Path to the processed corpus')
args = parser.parse_args()

dataset_path=args.dataset_path
tokenization_path=args.tokenization_path

for folder in ['train', 'validation']:
    filename = f'{tokenization_path}/{folder}.txt'
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
