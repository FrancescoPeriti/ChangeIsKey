import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from data_handler import DataHandler
from itertools import combinations, islice
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import gmean, entropy, spearmanr
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc as auc_score

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


def accuracy(y: np.array, y_true: np.array) -> tuple:
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


def spearman(y: np.array, y_true: np.array) -> tuple:
    """
    Calculates the spearman rank score for a ranking problem.
    Returns the correlation score and pvalues rounded to 3 decimal places.

    Args:
        y(np.array): array containing predicted values
        y_true(np.array): array containing ground truth values.
    Returns:
        corr, pvalue
    """
    corr, pvalue = spearmanr(y, y_true)
    return round(float(corr), 3), round(float(pvalue), 3)


def apd(E1: torch.tensor, E2: torch.tensor, metric: str = 'cosine') -> float:
    '''
    Average Pairwise Distance. Calculates the APD between two embedding matrix E1 and E2 using the given distance metric.

    Args:
        E1(torch.tensor): first embedding matrix
        E2(torch.tensor): second embedding matrix
        metric(str, default='cosine'): metric to use
    Returns:
        apd
    '''
    E1 = E1[torch.randint(E1.shape[0], (100,))]
    E2 = E2[torch.randint(E2.shape[0], (100,))]
    return cdist(E1, E2, metric=metric).mean()


def prt(E1: torch.tensor, E2: torch.tensor, metric: str = 'cosine') -> float:
    '''
    Inverse Prototype Distance. Calculates the PRT between two embedding matrix E1 and E2 using the given distance metric.

    Args:
        E1(torch.tensor): first embedding matrix
        E2(torch.tensor): second embedding matrix
        metric(str, default='cosine'): metric to use
    Returns:
        prt
    '''
    return cdist(E1.mean(axis=0).unsqueeze(0), E2.mean(axis=0).unsqueeze(0), metric=metric)[0][0]


def lsc_measuring(E1: torch.tensor, E2: torch.tensor, metric: object) -> np.array:
    """
    Args:
        E1(torch.tensor): first embedding matrix
        E2(torch.tensor): second embedding matrix
        metric(object): function to use
    Returns:
        np.array of lsc scores
    """
    if E2 is not None:
        return np.array([metric(E1[word], E2[word]) for word in E1.keys()])
    else:
        return np.array([metric(E1[word]) for word in E1.keys()])


def cross_validation(y: np.array, y_true: np.array, train_sets: list, test_sets: list, graded: bool = True) -> tuple:
    """
    Computes cross-validation statistics for a prediction task.

    Args:
        y(np.array): The predicted values for each sample.
        y_true(np.array): The true values for each sample.
        train_sets(list): A list of indices defining the samples in each training set.
        test_sets(list): A list of indices defining the samples in each test set.
        graded(bool, default=True): If True, computes graded statistics (Spearman correlation and p-value) for each fold. If False, computes binary statistics (accuracy and threshold) for each fold.
    Returns:
        score_train, info_train, score_test, info_test
        When graded is true:
         - 'score' is the average Spearman correlation coefficient between the predicted and true values for the test sets.
         - 'info' is the average corresponding pvalue
        When graded is False:
        - 'score' is the average accuracy between the predicted and true values for the test sets.
        - 'info' is the average corresponding classification threshold
    """

    if graded:
        metric = spearman
    else:
        metric = accuracy

    score_test, info_test = list(), list()
    score_train, info_train = list(), list()
    for i in range(0, len(train_sets)):

        # -- Test set --
        score, info = metric(y[test_sets[i]], y_true[test_sets[i]])
        score_test.append(score)
        info_test.append(info)

        # -- Train set --
        score, info = metric(y[train_sets[i]], y_true[train_sets[i]])
        score_train.append(score)
        info_train.append(info)

    score_train, info_train = np.array(score_train).mean(), np.array(info_train).mean()
    score_test, info_test = np.array(score_test).mean(), np.array(info_test).mean()

    return score_train, info_train, score_test, info_test


def permutation_test(y:np.array, y_true:np.array, n_resamples:int=1000) -> float:
    """
    Perform a permutation test to assess the significance of the Spearman correlation coefficient between `y` and `y_true`.

    Args:
        y(array-like): A one-dimensional array of values to be correlated with `y_true`.
        y_true(array-like): A one-dimensional array of true values with which `y` is to be correlated.
        n_resamples(int, default=1000): The number of resamples to use for the permutation test.

    Returns:
        pvalue(float): The p-value for the permutation test, rounded to 3 decimal places.
    """

    def spearmanr_statistic(y):  # permute only `x`
        return spearmanr(y, y_true).correlation

    res_exact = stats.permutation_test((y,), spearmanr_statistic, n_resamples=n_resamples,
                                       permutation_type='pairings')

    return res_exact.pvalue.round(3)


def prototype_distance_matrix(E1: dict, E2: dict) -> dict:
    """
    Calculates the cosine distance between two sets of word embeddings.

    Args:
        E1 (dict): A first dictionary of word embeddings for a specific set of words.
        E2 (dict): A dictionary of word embeddings for the same set of words.

    Returns:
        dict: A dictionary of cosine distances between the word embeddings of the two sets of words.
    """

    dist = dict()
    for word in E1.keys():
        dist[word] = cdist(E1[word], E2[word], metric='cosine')
    return dist


class BruteForce:
    def __init__(self, data_folder: str, dataset: str, model: str, layers: int = 12, name: str = None):
        self.dh = DataHandler(data_folder)
        self.dataset = dataset
        self.model = model
        self.name = name
        self.layers = layers

    def evaluate_mix_embeddings(self, depth: int = 3) -> list:
        """
        Evaluate the performance of mixed embeddings (extracted from different layers)
        with different combinations and aggregation methods.

        Args:
            depth (int, default=3): the size of the possible combinations

        Returns:
            a list of dictionaries containing the evaluation results for each combination and aggregation method.
            Each dictionary contains the following keys:
            - idx (str): the combination idx.
            - agg (str): the aggregation method used for the combination of .
            - corr (float): the Spearman correlation between the predicted and true scores.
            - pvalue (float): the p-value associated with the Spearman correlation.
            - acc (float): the accuracy of the predicted labels.
            - thr (float): the threshold used to generate the predicted labels.
            - auc_roc (float): the area under the ROC curve for the predicted labels.
            - auc_pr (float): the area under the precision-recall curve for the predicted labels.
            - changed_corr (float): the Spearman correlation between the predicted and true scores for the changed words only.
            - changed_pvalue (float): the p-value associated with the Spearman correlation for the changed words only.
            - stable_corr (float): the Spearman correlation between the predicted and true scores for the stable words only.
            - stable_pvalue (float): the p-value associated with the Spearman correlation for the stable words only.
            - measure (str): the name of the performance measure used (either "apd" or "prt").
            - score (np.array): predicted scores with the idx combination
        """

        # combinations
        combs = list()
        for i in range(2, depth + 1):
            combs += list(combinations(list(range(1, self.layers + 1)), i))

        # ground truth
        binary = self.dh.load_binary(self.dataset, self.name).score.values
        graded = self.dh.load_graded(self.dataset, self.name).score.values

        # mask
        mask_stable, mask_changed = list(), list()
        for i, k in enumerate(binary):
            if not k:
                mask_stable.append(i)
            else:
                mask_changed.append(i)

        E1, E2 = self.dh.load_all_embeddings(self.dataset, self.model, self.layers, self.name)

        # Results
        res = list()
        for comb in tqdm(combs, desc='Combining embeddings'):
            for agg in ['mean', 'concat']:
                comb_name = '-'.join([f'L{l}' for l in comb])

                # self.dh.load_mix_embeddings(self.dataset, self.model, comb, agg)
                E1_mix, E2_mix = self.dh.mix_embeddings(E1, E2, comb, agg)

                # all apd experiments are on the same set
                set_seed(SEED)

                for k, v in {'apd': apd, 'prt': prt}.items():
                    y = lsc_measuring(E1_mix, E2_mix, v)

                    # full evaluation
                    corr, pvalue = spearman(y, graded)
                    acc, thr = accuracy(y, binary)
                    labels = np.array([1 if i >= thr else 0 for i in graded])

                    # one label
                    if np.unique(labels).shape[0] != 2:
                        auc_roc = 0
                        auc_pr = 0
                    else:
                        auc_roc = roc_auc_score(labels, binary)
                        precision, recall, thresholds = precision_recall_curve(labels, binary)
                        auc_pr = auc_score(recall, precision)

                    # changed evaluation
                    changed_corr, changed_pvalue = spearman(y[mask_changed], graded[mask_changed])

                    # stable evaluation
                    stable_corr, stable_pvalue = spearman(y[mask_stable], graded[mask_stable])

                    res.append(dict(idx=comb_name, agg=agg,
                                    corr=corr, pvalue=pvalue,
                                    acc=acc, thr=thr,
                                    auc_roc=auc_roc, auc_pr=auc_pr,
                                    changed_corr=changed_corr, changed_pvalue=changed_pvalue,
                                    stable_corr=stable_corr, stable_pvalue=stable_pvalue,
                                    score=y,
                                    measure=k
                                    ))
        return res

    def evaluate_mix_measure(self, depth: int = 3, standardize: bool = True) -> list:
        """
        Evaluate the performance of mixed measures with different aggregation methods.

        Args:
            depth (int, default=3): the size of the possible combinations
            standardize (bool, default=True): If True, score are standardized

        Returns:
            a list of dictionaries containing the evaluation results for each combination and aggregation method.
            Each dictionary contains the following keys:
            - idx (str): the combination idx.
            - agg (str): the aggregation method used for the combination of .
            - corr (float): the Spearman correlation between the predicted and true scores.
            - pvalue (float): the p-value associated with the Spearman correlation.
            - acc (float): the accuracy of the predicted labels.
            - thr (float): the threshold used to generate the predicted labels.
            - auc_roc (float): the area under the ROC curve for the predicted labels.
            - auc_pr (float): the area under the precision-recall curve for the predicted labels.
            - changed_corr (float): the Spearman correlation between the predicted and true scores for the changed words only.
            - changed_pvalue (float): the p-value associated with the Spearman correlation for the changed words only.
            - stable_corr (float): the Spearman correlation between the predicted and true scores for the stable words only.
            - stable_pvalue (float): the p-value associated with the Spearman correlation for the stable words only.
            - score (np.array): predicted scores with the idx combination
        """

        targets = self.dh.load_targets(self.dataset, self.name).word.values
        binary = self.dh.load_binary(self.dataset, self.name).score.values
        graded = self.dh.load_graded(self.dataset, self.name).score.values

        # -- Prototype embeddings --
        PE1, PE2 = self.dh.load_all_prototype_embeddings(self.dataset, self.model, self.layers, self.name)

        # -- Distance matrix between embeddings --
        proto_dist = prototype_distance_matrix(PE1, PE2)

        new_measures = dict()

        # Prototype Hausdorff Distance
        new_measures['ProtoHD'] = lsc_measuring(PE1, PE2, lambda x1, x2: directed_hausdorff(x1, x2)[0])

        # Frobenius Norm
        new_measures['Fnorm'] = lsc_measuring(proto_dist, None, lambda x: np.linalg.norm(x, 'fro'))

        # Condition Number
        new_measures['Cond'] = lsc_measuring(proto_dist, None, lambda x: np.linalg.cond(x, 'fro'))

        # ERank
        new_measures['Erank'] = lsc_measuring(proto_dist, None,
                                              lambda x: np.exp(entropy(PCA().fit(x).singular_values_)))

        # Average of PRT over different layers
        new_measures['AvgPRT'] = lsc_measuring(proto_dist, None, lambda x: np.diag(x).mean())

        set_seed(SEED)
        for layer in range(1, self.layers + 1):
            y = self.dh.load_scores(self.dataset, self.model, layer)

            # old measures
            new_measures[f'APD{layer}'] = y[y['measure'] == 'apd_cosine'].score.values
            new_measures[f'PRT{layer}'] = y[y['measure'] == 'prt'].score.values
            new_measures[f'HD{layer}'] = y[y['measure'] == 'hd'].score.values

            # new measures with different distance matrix
            #E1, E2 = self.dh.load_embeddings(self.dataset, self.model, layer, self.name)

            # random pick words
            #E1 = {word: E1[word][torch.randint(E1[word].shape[0], (100,))] for word in E1}
            #E2 = {word: E2[word][torch.randint(E2[word].shape[0], (100,))] for word in E2}

            #distance_matrix = {word: cdist(E1[word], E2[word], metric='cosine') for word in E1}

            #new_measures[f'Fnorm{layer}'] = lsc_measuring(distance_matrix, None, lambda x: np.linalg.norm(x, 'fro'))
            #new_measures[f'Erank{layer}'] = lsc_measuring(distance_matrix, None,
            #                                              lambda x: np.exp(entropy(PCA().fit(x).singular_values_)))

        # additional info
        new_measures['word'] = targets
        new_measures['change'] = binary

        tmp = new_measures.copy()

        # measure to mix
        measures = set([i for i in tmp.keys() if i not in ['word', 'change']])

        # standardize
        if standardize:
            for m in measures:
                tmp[m] = (tmp[m] - tmp[m].mean()) / tmp[m].std()

        # combinations
        combs = list()
        for i in range(2, depth + 1):
            combs += list(combinations(measures, i))

        for comb in combs:
            comb_name = 'gmean ' + '-'.join(comb)
            tmp[comb_name] = gmean([tmp[m] for m in comb], axis=0)
            comb_name = 'mean ' + '-'.join(comb)
            tmp[comb_name] = np.mean([tmp[m] for m in comb], axis=0)

        # mask
        mask_changed = [i for i, k in enumerate(binary) if k]
        mask_stable = [i for i, k in enumerate(binary) if not k]

        # results
        res = list()
        for comb_name in tqdm(list(tmp.keys()), desc='Combining measures'):
            if comb_name in ['word', 'score']: continue

            # get aggregation
            agg = comb_name.split()[0]

            # score
            y = tmp[comb_name]

            # full evaluation
            try:
                corr, pvalue = spearman(y, graded)
                acc, thr = accuracy(y, binary)
                labels = np.array([1 if i >= thr else 0 for i in graded])

                auc_roc = roc_auc_score(labels, binary)
                precision, recall, thresholds = precision_recall_curve(labels, binary)
                auc_pr = auc_score(recall, precision)
            except:
                # something wrong with nan values
                continue

            # changed evaluation
            changed_corr, changed_pvalue = spearman(y[mask_changed], graded[mask_changed])

            # stable evaluation
            stable_corr, stable_pvalue = spearman(y[mask_stable], graded[mask_stable])

            res.append(dict(idx=comb_name, agg=agg, corr=corr, pvalue=pvalue,
                            acc=acc, thr=thr, auc_roc=auc_roc, auc_pr=auc_pr,
                            changed_corr=changed_corr, changed_pvalue=changed_pvalue,
                            stable_corr=stable_corr, stable_pvalue=stable_pvalue, score=y))

        return res

    def validation(self, res:list, topn:int=100,
                   n_combinations: int = 1000, percent: int = 40):

        set_seed(SEED)
        binary = self.dh.load_binary(self.dataset, self.name).score.values
        graded = self.dh.load_graded(self.dataset, self.name).score.values
        mask_changed = [i for i, k in enumerate(binary) if k]
        mask_stable = [i for i, k in enumerate(binary) if not k]

        if 'Russian' not in self.dataset:
            ascending=False
        else:
            ascending=True

        res = pd.DataFrame(res)
        res = res.sort_values('corr', ascending=ascending).reset_index(drop=True).head(topn)

        # -- Train and test sets --
        n_changed_words = binary[mask_changed].shape[0]
        n_changed_words_per_test_set = int(n_changed_words * percent / 100)
        
        changed_test_sets = [list(i) for i in
                             islice(combinations(list(range(0, n_changed_words)), n_changed_words_per_test_set),
                                    n_combinations)]
        changed_train_sets = [[i for i in range(0, n_changed_words) if i not in t] for t in changed_test_sets]

        
        # results
        new_res = dict(pt_pvalue=list(), stable_pt_pvalue=list(), changed_pt_pvalue=list(),
                       changed_corr_test=list(), changed_corr_train=list(),
                       changed_pvalue_test=list(), changed_pvalue_train=list())
        for i, row in tqdm(res.iterrows(), total=topn):
            # permutation test
            pt_pvalue = permutation_test(row['score'], graded)
            changed_pt_pvalue = permutation_test(row['score'][mask_changed], graded[mask_changed])
            stable_pt_pvalue = permutation_test(row['score'][mask_stable], graded[mask_stable])

            # cross validation test
            changed_corr_train, changed_pvalue_train, changed_corr_test, changed_pvalue_test = cross_validation(
                row['score'][mask_changed], graded[mask_changed], changed_train_sets, changed_test_sets)

            new_res['pt_pvalue'].append(pt_pvalue)
            new_res['stable_pt_pvalue'].append(stable_pt_pvalue)
            new_res['changed_pt_pvalue'].append(changed_pt_pvalue)
            new_res['changed_corr_test'].append(changed_corr_test)
            new_res['changed_corr_train'].append(changed_corr_train)
            new_res['changed_pvalue_test'].append(changed_pvalue_test)
            new_res['changed_pvalue_train'].append(changed_pvalue_train)

        for k in new_res:
            res[k] = new_res[k]

        res['score'] = [arr.tolist() for arr in res['score']]
        
        return res


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog='Brute force experiments', add_help=True)
    parser.add_argument('-f', '--main_folder',
                        type=str,
                        help='Project folder where all the data are stored.')
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Name of the dataset over which performing brute force experiment')
    parser.add_argument('-m', '--model',
                        type=str,
                        help='Name of the bert model from which the embeddings are extracted.')
    parser.add_argument('-l', '--layers',
                        type=int, default=12,
                        help='Number of layers of the model')
    parser.add_argument('-n', '--name',
                        type=str, default=None,
                        help='Suffix of the truth file to use for evaluation.')
    parser.add_argument('-D', '--depth',
                        type=int, default=3,
                        help='Maximum size of the tried combinations')
    parser.add_argument('-o', '--output_folder',
                        type=str, default='',
                        help='Name of the output folder where results will be stored')
    args = parser.parse_args()

    b = BruteForce(args.main_folder, args.dataset, args.model, args.layers, args.name)
    #res = b.evaluate_mix_embeddings(args.depth)
    #res = b.validation(res)

    out_folder = f'{args.output_folder}/LSC/{args.dataset}/{args.model}'
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    #res.to_csv(f'{out_folder}/bf_mix_embeddings.txt', index=False, sep='\t')

    res = b.evaluate_mix_measure(args.depth, standardize=False)
    res = b.validation(res)
    res.to_csv(f'{out_folder}/bf_mix_measures.txt', index=False, sep='\t')

    res = b.evaluate_mix_measure(args.depth, standardize=True)
    res = b.validation(res)
    res.to_csv(f'{out_folder}/bf_mix_std_measures.txt', index=False, sep='\t')
