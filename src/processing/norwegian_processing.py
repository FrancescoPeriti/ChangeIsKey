import os
from pathlib import Path
import pandas as pd
import argparse


parser = argparse.ArgumentParser(prog='Processing NorDiaChange-Norwegian',
                                 description='Remove pos tags')
parser.add_argument('-d', '--dataset_folder',
                    type=str, help='Folder of the processed dataset')
parser.add_argument('-t', '--tokenization_folder',
                    type=str, help='Folder of the tokenization output')
args = parser.parse_args()

repo = 'nor_dia_change'
tokenization_folder = args.tokenization_folder
dataset_folder = args.dataset_folder

c = 0
subset1 = dict(corpus1=list(), corpus2=list())
subset2 = dict(corpus1=list(), corpus2=list())

for k,v in {'subset1': (1929, 1970), 'subset2': (1980, 2012)}.items():
    repo_folder = f'{repo}/{k}/data'
    corpus1, corpus2 = list(), list()
    for target in os.listdir(repo_folder):
        filename=f'{repo_folder}/{target}/uses.csv'
        data=pd.read_csv(filename, sep='\t')

        for i, row in data.iterrows():

            # bad annotation: check the data
            if target == 'sete' and 'september septett' in row['context']:
                continue
            
            start, end = row['indexes_target_token'].split(':')
            record = dict(sentidx=c, lemma=target,
                                token=row['lemma'],
                                start=int(start), end=int(end),
                                sent=row['context'])
            c+=1
            
            if row['date'] in [1970, 1980]:
                corpus1.append(record)
            else:
                corpus2.append(record)
            
            if row['date'] == 1929:
                subset1['corpus1'].append(row['context']+'\n')
            elif row['date'] == 1970:
                subset1['corpus2'].append(row['context']+'\n')

            if row['date'] == 1980:
                subset2['corpus1'].append(row['context']+'\n')
            elif row['date'] == 2012:
                subset2['corpus2'].append(row['context']+'\n')

        folder = tokenization_folder.replace("NorDiaChange", "NorDiaChange12" if k == 'subset1' else "NorDiaChange23")

        for corpus in ['corpus1', 'corpus2']:
            Path(f'{folder}/{corpus}/token').mkdir(parents=True, exist_ok=True)
            output=f'{folder}/{corpus}/token/{target}.txt'

            df=pd.DataFrame(eval(corpus))
            df.to_json(output, orient='records', lines=True)

for k in ['subset1', 'subset2']:
    folder = dataset_folder.replace("NorDiaChange", "NorDiaChange12" if k == 'subset1' else "NorDiaChange23")

    for corpus in ['corpus1', 'corpus2']:
        Path(f'{folder}/{corpus}/token/').mkdir(parents=True, exist_ok=True)
        with open(f'{folder}/{corpus}/token/{corpus}.txt',
              mode='w+', encoding='utf-8') as f:
            f.writelines(eval(k)[f'{corpus}'])
