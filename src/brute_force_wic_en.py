import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from scipy.stats import gmean
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score, roc_curve
warnings.filterwarnings("ignore")


def load_data(dataset, model, layers=12, dataset_path="datasets", embedding_path="embeddings"):
    # Train and Test set
    if dataset == 'WiC-English':
        train, test = 'dev', 'test'
    else:
        train, test = 'train', 'test'

    # Load gold data
    path_gold_train = f'{dataset_path}/{dataset}/{train}/{train}.gold.txt'
    path_gold_test = f'{dataset_path}/{dataset}/{test}/{test}.gold.txt'
    Y_train = pd.read_csv(path_gold_train, sep='\t', names=['gold']).values
    Y_test = pd.read_csv(path_gold_test, sep='\t', names=['gold']).values

    # Load embedding - Train set
    embs_train = load_embs(dataset, train, model, layers, embedding_path=embedding_path)
    embs_test = load_embs(dataset, test, model, embedding_path=embedding_path)

    # Compute basis scores
    scores_train = compute_scores(embs_train)
    scores_test = compute_scores(embs_test)

    return dict(embs_train=embs_train, embs_test=embs_test,
                scores_train=scores_train, scores_test=scores_test,
                Y_train=Y_train, Y_test=Y_test)

def load_embs(dataset, folder, model, layers=12, embedding_path="embeddings"):
    embeddings = defaultdict(dict)

    for l in range(1, layers + 1):
        f = f'{embedding_path}/{dataset}/{folder}/{model}/{l}.pt'
        E = torch.load(f)
        E1, E2 = list(), list()
        for i in range(0, E.shape[0], 2):
            E1.append(E[i])
            E2.append(E[i + 1])

        embeddings['sent1'][l] = torch.stack(E1)
        embeddings['sent2'][l] = torch.stack(E2)

    return embeddings

def compute_scores(embeddings, layers=12):
    scores = defaultdict(list)

    n_pairs = embeddings['sent1'][1].shape[0]
    for i in range(n_pairs):
        # Embeddings wrapper: embeddings from all layers for sent1 and sent2
        embs_t1, embs_t2 = list(), list()

        for j in range(1, layers + 1):
            # embs from layer j for sent1 and sent2
            embs_t1_lj, embs_t2_lj = embeddings['sent1'][j][i], embeddings['sent2'][j][i]
            embs_t1.append(embs_t1_lj)
            embs_t2.append(embs_t2_lj)

            # Cosine Distance (CD) and Similarity (CS)
            cd = cdist([embs_t1_lj.numpy()], [embs_t2_lj.numpy()], metric='cosine')[0][0]
            scores[f'CD{j}'].append(cd)
            scores[f'CS{j}'].append(1 - cd)

        # Cosine Distance and Similarity Matrix between embeddings of different layers
        cd_matrix = cdist([e.numpy() for e in embs_t1], [e.numpy() for e in embs_t2], metric='cosine')
        cs_matrix = 1 - cdist([e.numpy() for e in embs_t1], [e.numpy() for e in embs_t2], metric='cosine')
        cd_matrix_lowlev = cd_matrix[:4, :4]
        cs_matrix_lowlev = cs_matrix[:4, :4]
        cd_matrix_medlev = cd_matrix[4:8, 4:8]
        cs_matrix_medlev = cs_matrix[4:8, 4:8]
        cd_matrix_highlev = cd_matrix[-4:, -4:]
        cs_matrix_highlev = cs_matrix[-4:, -4:]

        # Average CD over multiple layers
        scores['AvgCD'].append(np.diag(cd_matrix).mean())
        scores['AvgCS'].append(np.diag(cs_matrix).mean())

        # Frobinius Norm
        scores['FnormCD'].append(np.linalg.norm(cd_matrix, 'fro'))
        scores['FnormCS'].append(np.linalg.norm(cs_matrix, 'fro'))
        scores['-FnormCD'].append(np.linalg.norm(-cd_matrix, 'fro'))
        scores['-FnormCS'].append(np.linalg.norm(-cs_matrix, 'fro'))
        #scores['FnormCDlow'].append(np.linalg.norm(cd_matrix_lowlev, 'fro'))
        #scores['FnormCSlow'].append(np.linalg.norm(cs_matrix_lowlev, 'fro'))
        #scores['FnormCDmed'].append(np.linalg.norm(cd_matrix_medlev, 'fro'))
        #scores['FnormCSmed'].append(np.linalg.norm(cs_matrix_medlev, 'fro'))
        scores['FnormCDhigh'].append(np.linalg.norm(cd_matrix_highlev, 'fro'))
        scores['FnormCShigh'].append(np.linalg.norm(cs_matrix_highlev, 'fro'))

        # Cond
        scores['CondCD'].append(np.linalg.cond(cd_matrix, 'fro'))
        scores['CondCS'].append(np.linalg.cond(cs_matrix, 'fro'))
        scores['-CondCD'].append(-np.linalg.cond(cd_matrix, 'fro'))
        scores['-CondCS'].append(-np.linalg.cond(cs_matrix, 'fro'))
        scores['CondCDlow'].append(np.linalg.cond(cd_matrix_lowlev, 'fro'))
        scores['CondCSlow'].append(np.linalg.cond(cs_matrix_lowlev, 'fro'))
        #scores['CondCDmed'].append(np.linalg.cond(cd_matrix_medlev, 'fro'))
        #scores['CondCSmed'].append(np.linalg.cond(cs_matrix_medlev, 'fro'))
        scores['CondCDhigh'].append(np.linalg.cond(cd_matrix_highlev, 'fro'))
        scores['CondCShigh'].append(np.linalg.cond(cs_matrix_highlev, 'fro'))
        scores['-CondCDlow'].append(-np.linalg.cond(cd_matrix_lowlev, 'fro'))
        scores['-CondCSlow'].append(-np.linalg.cond(cs_matrix_lowlev, 'fro'))
        #scores['-CondCDmed'].append(-np.linalg.cond(cd_matrix_medlev, 'fro'))
        #scores['-CondCSmed'].append(-np.linalg.cond(cs_matrix_medlev, 'fro'))
        scores['-CondCDhigh'].append(-np.linalg.cond(cd_matrix_highlev, 'fro'))
        scores['-CondCShigh'].append(-np.linalg.cond(cs_matrix_highlev, 'fro'))

        # Hausdorf Distance and Similarity
        E1, E2 = [e.numpy() for e in embs_t1], [e.numpy() for e in embs_t2]
        hd = directed_hausdorff(E1, E2)[0]
        hd_low = directed_hausdorff(E1[:4], E2[:4])[0]
        hd_med = directed_hausdorff(E1[4:8], E2[4:8])[0]
        hd_high = directed_hausdorff(E1[-4:], E2[-4:])[0]
        scores['HD'].append(hd)
        scores['HS'].append(1 / (0.001 + hd))
        #scores['HDlow'].append(hd_low)
        #scores['HSlow'].append(1 / (0.001 + hd_low))
        #scores['HDmed'].append(hd_med)
        #scores['HSmed'].append(1 / (0.001 + hd_med))
        scores['HDhigh'].append(hd_high)
        scores['HShigh'].append(1 / (0.001 + hd_high))

        # Average Pairwise Distance and Similarity
        cd = cdist(E1, E2, metric='cosine')
        cd_low = cdist(E1[:4], E2[:4], metric='cosine')
        cd_med = cdist(E1[4:8], E2[4:8], metric='cosine')
        cd_high = cdist(E1[-4:], E2[-4:], metric='cosine')
        scores['APD'].append(cd.mean())
        scores['APS'].append((1 - cd).mean())
        #scores['APDlow'].append(cd_low.mean())
        #scores['APSlow'].append((1 - cd_low).mean())
        #scores['APDmed'].append(cd_med.mean())
        #scores['APSmed'].append((1 - cd_med).mean())
        scores['APDhigh'].append(cd_high.mean())
        scores['APShigh'].append((1 - cd_high).mean())

    for s in scores:
        scores[s] = np.array(scores[s])

    return scores

def eval_scores(Y_train, Y_test, scores_train, scores_test):
    acc_train, thr = top_accuracy(Y_train, scores_train)
    
    labels_test = [int(i >= thr) for i in scores_test]
    labels_train = [int(i >= thr) for i in scores_train]
    
    acc_test = accuracy_score(Y_test, labels_test)
    f_score_train = f1_score(Y_train, labels_train)
    f_score_test = f1_score(Y_test, labels_test)
    pr_test = precision_score(Y_test, labels_test)
    pr_train = precision_score(Y_train, labels_train)
    recall_test = recall_score(Y_test, labels_test)
    recall_train = recall_score(Y_train, labels_train)
    return dict(#scores_train=scores_train.tolist(), scores_test=scores_test.tolist(),
                acc_train=acc_train, acc_test=acc_test,
                f_score_train=f_score_train, f_score_test=f_score_test,
                pr_train=pr_train, pr_test=pr_test,
                recall_train=recall_train, recall_test=recall_test, thr=thr)

def brute_force(combs, shared_list, Y_train, Y_test, scores_train, scores_test, standardize_=True):
    if standardize_:
        for m in scores_train:
            scores = standardize(np.array(scores_train[m].tolist()+scores_test[m].tolist()))
            scores_train[m] = scores[:scores_train[m].shape[0]]
            scores_test[m] = scores[scores_train[m].shape[0]:]

    for comb in tqdm(combs, position=0, leave=True):
        # Gmean (is not suitable for negative values)
        if not standardize_:
            train_values = gmean([scores_train[m] for m in comb], axis=0)
            test_values = gmean([scores_test[m] for m in comb], axis=0)
            if not np.isnan(train_values).any() and not np.isnan(test_values).any():
                res = eval_scores(Y_train, Y_test, train_values, test_values)
                res['agg'] = 'gmean'
                res['comb'] = comb
                shared_list.append(res)

        # Mean
        train_values = np.mean([scores_train[m] for m in comb], axis=0)
        test_values = np.mean([scores_test[m] for m in comb], axis=0)

        res = eval_scores(Y_train, Y_test, train_values, test_values)
        res['agg'] = 'mean'
        res['comb'] = comb
        shared_list.append(res)

        # Max
        train_values = np.max([scores_train[m] for m in comb], axis=0)
        test_values = np.max([scores_test[m] for m in comb], axis=0)
        res = eval_scores(Y_train, Y_test, train_values, test_values)
        res['agg'] = 'max'
        res['comb'] = comb
        shared_list.append(res)

        # - Linear regression
        X_train = np.stack([scores_train[m] for m in comb]).T
        X_test = np.stack([scores_test[m] for m in comb]).T
        reg = LinearRegression().fit(X_train, Y_train)
        train_values = reg.predict(X_train)
        test_values = reg.predict(X_test)
        res = eval_scores(Y_train, Y_test, train_values, test_values)
        res['comb'] = comb
        res['agg'] = 'reg'
        shared_list.append(res)

    return shared_list

def list2chunks(combs) -> list:
    """Split a list into evenly sized chunks"""
    n = multiprocessing.cpu_count()
    k, m = divmod(len(combs), n)
    return (combs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def run(measures_list, depth, data, standardize_=False):
    """Perform tokenization in multiprocessing"""

    combs = list()
    for i in range(1, depth + 1):
        combs += list(combinations(measures_list, i))

    all_processes = list()
    shared_list = multiprocessing.Manager().list()
    for sub_list in list2chunks(combs):
        p = multiprocessing.Process(target=brute_force, args=(sub_list, shared_list,
                                                              data['Y_train'], data['Y_test'],
                                                              data['scores_train'], data['scores_test'],
                                                              standardize_))
        all_processes.append(p)
        p.start()

    for p in all_processes:
        p.join()

    shared_list = list(shared_list)
    return pd.DataFrame(shared_list).sort_values('acc_test', ascending=False)


def standardize(x):
    return (x - x.mean()) / x.std()


def top_accuracy(y_true: np.array, y: np.array) -> tuple:
    """
    Calculates the accuracy score for a binary classification problem.
    The function first calculates the False Positive Rate (FPR), True Positive Rate (TPR), and Thresholds using the
    roc_curve function from Scikit-learn. Next, it calculates the accuracy score for each threshold value and returns
    the maximum accuracy score and its corresponding threshold value rounded to 3 decimal places.

    Args:
        y(np.array): array containing predicted values
        y_true(np.array): array containing ground truth values.
    Returns:
        acc, thr
    """

    # False Positive Rate - True Positive Rate
    fpr, tpr, thresholds = roc_curve(y_true, y)

    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true, [m >= thresh for m in y]))

    accuracy_scores = np.array(accuracy_scores)

    # Max accuracy
    max_accuracy = accuracy_scores.max()

    # Threshold associated to the maximum accuracy
    max_accuracy_threshold = thresholds[accuracy_scores.argmax()]

    return round(float(max_accuracy), 3), max_accuracy_threshold


if __name__ == '__main__':

    dataset_path = '/mimer/NOBACKUP/groups/cik_data/datasets/WiC/'
    embedding_path = '/mimer/NOBACKUP/groups/cik_data/contextualized_embeddings/WiC/'

    layers = 12
    for model in ['bert-base-uncased', 'bert-base-multilingual-cased', 'cambridgeltl_mirrorwic-bert-base-uncased']:
        wic_en = load_data('WiC-English', model, dataset_path=dataset_path, embedding_path=embedding_path)
        #tempo_wic = load_data('TempoWiC', model, dataset_path=dataset_path, embedding_path=embedding_path)
        dwug_en = load_data('dwug_en', model, dataset_path=dataset_path, embedding_path=embedding_path)

        measures_list = list(wic_en['scores_train'].keys())
        output = '/mimer/NOBACKUP/groups/cik_data/brute_force/WiC'
        Path(f'{output}/WiC-English/{model}/').mkdir(parents=True, exist_ok=True)
        Path(f'{output}/dwug_en/{model}/').mkdir(parents=True, exist_ok=True)
        Path(f'{output}/TempoWiC/{model}/').mkdir(parents=True, exist_ok=True)

        print('WiC-English', model)
        res = run(measures_list, 3, wic_en, standardize_=False)
        res.to_csv(f'{output}/WiC-English/{model}/measures.txt', sep='\t')
        res = run(measures_list, 3, wic_en, standardize_=True)
        res.to_csv(f'{output}/WiC-English/{model}/std_measures.txt', sep='\t')

        #print('TempoWiC', model)
        #res = run(measures_list, 3, tempo_wic, standardize_=False)
        #res.to_csv(f'{output}/TempoWiC/{model}/measures.txt', sep='\t')
        #res = run(measures_list, 3, tempo_wic, standardize_=True)
        #res.to_csv(f'{output}/TempoWiC/{model}/std_measures.txt', sep='\t')

        print('dwug_en', model)
        res = run(measures_list, 3, wic_en, standardize_=False)
        res.to_csv(f'{output}/dwug_en/{model}/measures.txt', sep='\t')
        res = run(measures_list, 3, wic_en, standardize_=True)
        res.to_csv(f'{output}/dwug_en/{model}/std_measures.txt', sep='\t')
