import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from extraction import AttentionExtraction

parser = argparse.ArgumentParser(prog='Attention extraction',
                                 description='Extract the attention matrices for each sequence containing a target word')
parser.add_argument('-t', '--tokenized_dataset',
                    type=str,
                    help='A string representing the directory path to a tokenized dataset for LSC detection. '
                         'This dataset should contain pre-tokenized text for the target words.')
parser.add_argument('-m', '--model',
                    type=str, default='bert-base-uncased',
                    help='A string representing the name of the Hugging Face pre-trained model to use for attention extraction.')
parser.add_argument('-M', '--max_length',
                    type=int, default=512,
                    help='An integer representing the maximum sequence length to use for the attention extraction process. '
                         'Default value is 512.')
parser.add_argument('-l', '--layers',
                    type=int, default=12,
                    help='An integer representing the number of encoder layers of the pre-trained model to use for attention extraction. '
                         'Default value is 12.')
parser.add_argument('-b', '--batch_size',
                    type=int, default=8,
                    help='An integer representing the batch size to use for the attention extraction process. '
                         'Default value is 8.')
parser.add_argument('-o', '--output',
                    type=str,
                    help='A string representing the directory path to save the attention matrices.')
parser.add_argument('-n', '--sampling',
                    type=int, default=0,
                    help='An integer representing the number of sentences to sample randomly from the dataset. Default is 0 (no sampling).')
parser.add_argument('-T', '--targets',
                    type=str,
                    help='A string representing the directory path to a text file containing the target words to extract attention for.')
args = parser.parse_args()


# target words
words = [word.strip() for word in open(args.targets, mode='r', encoding='utf-8').readlines()]


a = AttentionExtraction(args.model)
for corpus in ['corpus1', 'corpus2']:
    for word in tqdm(words, desc=corpus):
        # i/o
        tokenization_input = f'{args.tokenized_dataset}/{corpus}/token/{word}.txt'
        attentions_output = f'{args.output}/{args.model.replace("/", "_")}/{corpus}/token'

        # extraction
        attentions = a.extract(dataset=tokenization_input, batch_size=args.batch_size,
                               max_length=args.max_length, layers=args.layers, sampling=args.sampling)

        # store attentions
        for l in range(1, args.layers + 1):
            Path(f'{attentions_output}/{l}/').mkdir(parents=True, exist_ok=True)
            attn_l = [a[l-1] for a in attentions]

            with open(f'{attentions_output}/{l}/{word}.pickle', 'wb') as f:
                pickle.dump(attn_l, f, protocol=pickle.HIGHEST_PROTOCOL)
