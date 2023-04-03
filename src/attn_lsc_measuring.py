import json
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.stats import entropy
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import Dataset

SEED = 42

def set_seed(seed: int):
    """
    This function sets the seed for the random number generators in Python's built-in random module, NumPy, PyTorch CPU, and PyTorch GPU.
    This is useful for ensuring reproducibility of results.

    Args:
        seed (int): The seed to set the random number generators to.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset(dataset: str) -> Dataset:
    """
    Loads a dataset from a file and returns it as a Dataset object.

    Args:
        dataset (str): A string representing the path to the dataset file.

    Returns:
        A Dataset object representing the loaded dataset.
    """
    rows = list()
    with open(dataset, mode='r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '': continue
            row = json.loads(line)
            if row is None: continue
            rows.append(row)

    return Dataset.from_list(rows)

def random_sampling(dataset: Dataset, n: int) -> Dataset:
    """
    This method randomly samples n instances from a given dataset.
    This is used to reduce the size of the dataset when processing large datasets with limited resources.

    Args:
        dataset(Dataset): The dataset to be sampled.
        n(int): An integer representing the number of sentences to be sampled.

    Returns:
        dataset
    """
    dataset.shuffle(seed=SEED)
    return dataset.select(range(0, min(dataset.num_rows, n)))

def gini(array:np.array) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    Based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    From: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    Args:
        array(np.array): input array

    Returns:
        gini index
    """

    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient


def lsc_leveraging_full_attn(targets:list, attentions_folder:str, model:str, layers:int=12) -> list:
    """
    Calculates the LSC score using measures based on whole attention matrices

    Args:
        targets(list): list of target words to analyze
        attentions_folder(list): folder containing the attention values
        model(str): name of the hugginface model where attention was extracted
        layers(int, default=12): number of layers of the considered model
    Returns:
        LSC scores per word
    """

    # -- Score wrapper --
    attn_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # -- Tested measures --
    measures = ['mean', 'max', 'median', 'entropy', 'std', 'wentropy', 'wlogentropy', 'gini']

    for word in tqdm(targets, position=0, leave=True, desc='Full attention'):
        for corpus in ['corpus1', 'corpus2']:
            for layer in range(1, layers + 1):
                A_path = f'{attentions_folder}/{model.replace("/", "_")}/{corpus}/token/{layer}/{word}.pickle'
                A = pickle.load(open(A_path, mode='rb'))

                # a -> attn_token_occurrence
                for a in A:
                    # head avg
                    a = a.mean(axis=1)
                    n_tokens = a.shape[1]

                    attn_stats['mean'][corpus][word][layer].append(a.mean().item())
                    attn_stats['std'][corpus][word][layer].append(a.std().item())
                    attn_stats['max'][corpus][word][layer].append(a.max().item())
                    attn_stats['median'][corpus][word][layer].append(a.median().item())
                    attn_stats['entropy'][corpus][word][layer].append(entropy(a.ravel()))
                    attn_stats['wentropy'][corpus][word][layer].append(entropy(a.ravel()) / n_tokens)
                    attn_stats['wlogentropy'][corpus][word][layer].append(entropy(a.ravel()) / math.log(n_tokens))
                    attn_stats['gini'][corpus][word][layer].append(gini(np.array(a.ravel())))

    # -- LSC scores --
    res = list()
    for m in measures:
        for word in targets:
            for layer in range(0, 12):
                A1 = torch.tensor(attn_stats[m]['corpus1'][word][layer]).mean().item()
                A2 = torch.tensor(attn_stats[m]['corpus2'][word][layer]).mean().item()
                res.append(dict(word=word, score=abs(A1 - A2), layer=layer + 1, measure=f'full_{m}'))

    return res


def lsc_leveraging_token_attn(targets: list, attentions_folder: str, tokenization_folder: str, model: str, layers: int = 12, sampling:int=0, max_length:int=512) -> list:
    """
    Calculates the LSC score using measures based on the token attention

    Args:
        targets(list): list of target words to analyze
        attentions_folder(list): folder containing the attention values
        tokenization_folder(list): folder containing the tokenized sentences containing the target word.
        model(str): name of the hugginface model where attention was extracted
        layers(int, default=12): number of layers of the considered model
        sampling(int, default=0): an integer representing the number of sentences to sample randomly from the dataset. Default is 0 (no sampling).
    Returns:
        LSC scores per word
    """

    # -- tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model)

    # -- LSC measures --
    measures = ['attn_to_others', 'attn_from_others', 'attn_token',
                'gini_attn_to_others', 'gini_attn_from_others', 'gini_attn_token',
                'entropy_attn_to_others', 'entropy_attn_from_others', 'entropy_attn_token',
                'wentropy_attn_to_others', 'wentropy_attn_from_others', 'wentropy_attn_token',
                'wlogentropy_attn_to_others', 'wlogentropy_attn_from_others', 'wlogentropy_attn_token',
                'mean_entropy_attn_token', 'mean_wentropy_attn_token']

    # -- LSC wrapper --
    lsc_attn_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # -- seed --
    set_seed(SEED)

    for word in tqdm(targets, desc='Token attention'):
        for corpus in ['corpus1', 'corpus2']:

            # -- Load dataset --
            dataset = load_dataset(f'{tokenization_folder}/{corpus}/token/{word}.txt')
            if sampling:
                dataset = random_sampling(dataset, sampling)

            for layer in range(1, layers + 1):
                attn_filename = f'{attentions_folder}/{model.replace("/", "_")}/{corpus}/token/{layer}/{word}.pickle'
                # Attention matrix
                A = pickle.load(open(attn_filename, mode='rb'))

                #print('N attn found:', len(A), 'num rows found:', dataset.num_rows)

                for i, row in enumerate(dataset):
                    #print(i, '-th sentence ---')
                    sent = row['sent']
                    start, end = row['start'], row['end']

                    sub_tokens = tokenizer.tokenize(sent[start:end])
                    left_tokens = tokenizer.tokenize(sent[:start])
                    
                    start_tokens = len(left_tokens)+1 #+1 for [CLS]
                    end_tokens = start_tokens + len(sub_tokens)

                    # Position of the target word (sub-tokens) in the i-th sentence
                    idx_sub_tokens = list(range(start_tokens, end_tokens))

                    #print('Shape:', A[i].shape, 'Num of tokens:', 2+len(tokenizer.tokenize(sent)), 'Token pos:', idx_sub_tokens[0])

                    # Truncation
                    if idx_sub_tokens[0] >= max_length:
                        continue

                    a = A[i]

                    # Attention of token to others
                    attn_to = a[:, idx_sub_tokens, :]

                    # Attention of others to token
                    attn_from = a[:, :, idx_sub_tokens]

                    # Average the attention of different heads
                    attn_to = attn_to.mean(axis=0)
                    attn_from = attn_from.mean(axis=0)

                    # Average the attention of different sub tokens
                    attn_to = attn_to.mean(axis=0)
                    attn_from = attn_from.mean(axis=1)
                    mean_attn = (attn_from + attn_to) / 2
                    n_tokens = attn_to.shape[0]

                    # Scores
                    lsc_attn_scores['attn_from_others'][corpus][word][layer].append(attn_from.mean(axis=0))
                    lsc_attn_scores['attn_to_others'][corpus][word][layer].append(attn_to.mean(axis=0))
                    lsc_attn_scores['gini_attn_from_others'][corpus][word][layer].append(mean_attn.mean(axis=0))
                    lsc_attn_scores['attn_token'][corpus][word][layer].append((attn_from + attn_to).mean(axis=0))

                    lsc_attn_scores['gini_attn_from_others'][corpus][word][layer].append(gini(np.array(attn_from)))
                    lsc_attn_scores['gini_attn_to_others'][corpus][word][layer].append(gini(np.array(attn_to)))
                    lsc_attn_scores['gini_attn_token'][corpus][word][layer].append(gini(np.array(mean_attn)))

                    lsc_attn_scores['entropy_attn_to_others'][corpus][word][layer].append(
                        entropy(np.array(attn_from)))
                    lsc_attn_scores['entropy_attn_from_others'][corpus][word][layer].append(
                        entropy(np.array(attn_to)))
                    lsc_attn_scores['entropy_attn_token'][corpus][word][layer].append(entropy(np.array(mean_attn)))

                    lsc_attn_scores['wentropy_attn_to_others'][corpus][word][layer].append(
                        entropy(np.array(attn_from)) / n_tokens)
                    lsc_attn_scores['wentropy_attn_from_others'][corpus][word][layer].append(
                        entropy(np.array(attn_to)) / n_tokens)
                    lsc_attn_scores['wentropy_attn_token'][corpus][word][layer].append(
                        entropy(np.array(mean_attn)) / n_tokens)

                    lsc_attn_scores['wlogentropy_attn_to_others'][corpus][word][layer].append(
                        entropy(np.array(attn_from)) / math.log(n_tokens))
                    lsc_attn_scores['wlogentropy_attn_from_others'][corpus][word][layer].append(
                        entropy(np.array(attn_to)) / math.log(n_tokens))
                    lsc_attn_scores['wlogentropy_attn_token'][corpus][word][layer].append(
                        entropy(np.array(mean_attn)) / math.log(n_tokens))

                    lsc_attn_scores['mean_entropy_attn_token'][corpus][word][layer].append(
                        (entropy(np.array(attn_from)) + entropy(np.array(attn_to))) / 2)
                    lsc_attn_scores['mean_wentropy_attn_token'][corpus][word][layer].append(
                        (entropy(np.array(attn_from)) + entropy(np.array(attn_to))) / n_tokens / 2)

    # -- Results --
    res = list()
    for m in measures:
        for layer in range(1, layers + 1):
            for word in targets:
                attn = abs(np.mean(lsc_attn_scores[m]['corpus1'][word][layer]) - np.mean(
                    lsc_attn_scores[m]['corpus2'][word][layer]))
                res.append(dict(word=word, layer=layer + 1, measure=f'token_{m}', score=attn))
    return res





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='clustering', add_help=True)
    parser.add_argument('-a', '--attentions',
                        type=str,
                        help='A string representing the directory path to the stored attentions for the benchmark dataset for LSC detection.')
    parser.add_argument('-T', '--tokenization',
                        type=str,
                        help='A string representing the directory path to the stored tokenized sentences.')
    parser.add_argument('-t', '--targets',
                        type=str,
                        help='A string representing the directory path to a text file containing the target words.')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='A string representing the output file.')
    parser.add_argument('-l', '--layers',
                        type=int, default=12,
                        help='An integer representing the number of encoder layers of the pre-trained model used for embedding extraction. '
                             'Default value is 12.')
    parser.add_argument('-m', '--model',
                        type=str, default='bert-base-uncased',
                        help='A string representing the name of the Hugging Face pre-trained model used for attention extraction.')
    parser.add_argument('-s', '--sampling', default=0,
                        type=int,
                        help='An integer representing the number of sentences to sample randomly from the dataset. Default is 0 (no sampling).')
    args = parser.parse_args()

    # -- Target --
    words = [word.strip() for word in open(args.targets, mode='r', encoding='utf-8').readlines()]

    # -- LSC scores by considering the attention of the full sentences --
    #res = lsc_leveraging_full_attn(words, args.attentions, args.model, layers=args.layers)
    #res = pd.DataFrame(res)
    #for l in range(1, args.layers+1):
    #    Path(f'{args.output}_full/{args.model.replace("/", "_")}/{l}').mkdir(parents=True, exist_ok=True)
    #    pd.DataFrame(res[res['layer']==l]).to_csv(f'{args.output}_full/{args.model.replace("/", "_")}/{l}/token.txt', sep='\t', index=False)

    # -- LSC scores by considering only the attenion of the target token --
    res = lsc_leveraging_token_attn(words, args.attentions, args.tokenization, args.model, args.layers, args.sampling)
    res = pd.DataFrame(res)
    for l in range(1, args.layers + 1):
        Path(f'{args.output}/{args.model.replace("/", "_")}/{l}').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res[res['layer'] == l]).to_csv(f'{args.output}/{args.model.replace("/", "_")}/{l}/token.txt', sep='\t',
                                                    index=False)
