import argparse
import importlib
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(prog='Tokenization script for processing LSC benchmark datasets.',
                                 add_help=True)
parser.add_argument('-d', '--dataset',
                    type=str,
                    help='Path to the directory containing the benchmark dataset for LSC detection. '
                    'This directory should include a file named \'targets.txt\' which contains a list of target words, and subdirectories for each corpus to be tokenized.')
parser.add_argument('-m', '--model',
                    type=str,
                    help='Name of the spaCy model to use for tokenization.')
parser.add_argument('-n', '--sampling',
                    type=int,default=0,
                    help='Number of sentences to randomly sample from the dataset. '
                    'Default value is 0, which means no sampling will be performed.')
parser.add_argument('-o', '--output',
                    type=str,
                    help='Path to the output directory where the selected sentences will be stored. '
                    'This directory will be created if it doesn\'t exist.')
parser.add_argument('-t', '--tokenization_class',
                    type=str, default='StandardSpacyTokenization',
                    help='Name of the Tokenization class to use for tokenization. '
                    'This should be the name of a class that extends the abstract base class Tokenization in the src.tokenization module.')
args = parser.parse_args()

# reflection -> get the class to instanziate
module = importlib.import_module(__name__)
tokenization_class = getattr(module, args.tokenization_class)

# target words
# target words
targets_filename = f'{args.dataset}/targets.txt' if not 'Russian' in args.dataset else f'{args.dataset}/targets_test.txt'
words = [word.strip() for word in open(targets_filename, mode='r', encoding='utf-8').readlines()]

# Tokenize the raw corpora
if args.model is not None:
    tokenizer = tokenization_class(args.model, words)
else:
    tokenizer = tokenization_class(words)
    
for corpus in range(1, 19):
    dataset_input = f'{args.dataset}/{corpus}.txt'
    tokenization_output = f'{args.output}/{corpus}/token'
    
    # run tokenization
    token_list = tokenizer.tokenize(dataset_input, args.sampling)
    
    # store results
    Path(tokenization_output).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(token_list)
    for word in words:
        filename = f'{tokenization_output}/{word}.txt'
        df[df['lemma'] == word].to_json(filename, orient='records', lines=True)
