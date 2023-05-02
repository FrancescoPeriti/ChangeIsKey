import spacy
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse

nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(prog='Processing TempoWiC')
parser.add_argument('-d', '--dataset_path', type=str, help='Path to the corpus to process')
parser.add_argument('-t', '--tokenization_path', type=str, help='Path to the processed corpus')
args = parser.parse_args()

dataset_path=args.dataset_path
tokenization_path=args.tokenization_path


for folder in ['dev', 'train', 'test']:
    dataset_path=args.dataset_path
    filename = f'{tokenization_path}/{folder}.txt'

    data = pd.read_csv(f'{dataset_path}/{folder}/{folder}.data.txt', sep='\t', names=['words', 'pos', 'indexes', 'sent1', 'sent2'])
    gold = pd.read_csv(f'{dataset_path}/{folder}/{folder}.gold.txt', sep='\t', names=['gold'])
    gold['gold'] = [int(i=='T') for i in gold['gold'].values]
    open(f'{dataset_path}/{folder}/{folder}.gold.txt', mode='w').writelines([str(i)+'\n' for i in gold['gold'].values])
    df = pd.concat([data, gold], axis=1)

    results = list()
    
    c = 0
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        word = row['words']
        for sent in ['sent1', 'sent2']:
            sent = row[sent]
        
            # Spacy errors
            if sent == 'She trod with care as the ground was slippery.':
                results.append(dict(sentidx=c, lemma='tread', token='trod', start=4, end=8, sent=sent.strip()))
                c+=1
                continue

            elif sent == "I 've got a house in the country .":
                sent = "I have got a house in the country ."

            elif sent == "The rising of the Holy Ghost .":
                results.append(dict(sentidx=c, lemma='rise', token='rising', start=4, end=10, sent=sent.strip()))
                c+=1
                continue
            
            elif sent == "Bases include oxides and hydroxides of metals and ammonia .":
                results.append(dict(sentidx=c, lemma='base', token='bases', start=0, end=5, sent=sent.strip()))
                c+=1
                continue
            
            start=-1
            for token in nlp(sent):
                if token.lemma_.lower() == word.lower():
                    start = token.idx
                    end = start+len(token.text)
                    break
                
            # Other spacy errors
            if start==-1:
                start=sent.find(word)
                end = start+len(sent[start:].split()[0])

            results.append(dict(sentidx=c, lemma=word, token=word, start=start, end=end, sent=sent.strip()))
            c+=1
    
    Path(f'/mimer/NOBACKUP/groups/cik_data/tokenization/WiC/WiC-English/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_json(filename, orient='records', lines=True)
