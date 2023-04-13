import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import f1_score
from itertools import combinations
from collections import defaultdict
from data_handler import DataHandler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy.stats import gmean, entropy, spearmanr
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc as auc_score

import warnings

warnings.filterwarnings("ignore")

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


def accuracy(y_true: np.array, y: np.array) -> tuple:
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


def permutation_test(y_true: np.array, y: np.array, n_resamples: int = 1000) -> float:
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
    def __init__(self, data_folder: str, dataset: str, model: str, layers: int = 12,
                 ignore_n_stable: int = 0, n_combinations: int = 100, percent: int = 40):
        """
        Args:
            data_folder(str): main folder containing the data of the project
            dataset(str): name of the dataset under consideration
            model(str): name of the bert-like model used
            layers(int, default=12): number of layers of the model
            ignore_n_stable(int, default=0): remove n stable words larger than the smallest changed words
            n_combinations(int, default=100): number of split for cross validation
            percent(int, default=40): percent of changed word for each split
        """

        self.dh = DataHandler(data_folder)
        self.dataset = dataset
        self.model = model
        self.name = None
        self.layers = layers
        self.ignore_n_stable = ignore_n_stable
        self.gt = self._load_ground_truth()

        self.changed_train_sets, self.changed_test_sets = self._set_split(binary=self.gt['binary'],
                                                                         mask=self.gt['mask_changed'],
                                                                         n_combinations=n_combinations, percent=percent)

    def _load_ground_truth(self) -> dict:
        """
        Load in memory ground truth scores, targets, and masks to access stable/changed words
        """

        # -- Load ground truth --
        binary = self.dh.load_binary(self.dataset, self.name).score.values
        graded = self.dh.load_graded(self.dataset, self.name).score.values
        targets = self.dh.load_targets(self.dataset, self.name).word.values

        # -- Remove the n stable words that changed most than changed words --
        targets, graded, binary, mask_stable, mask_changed, mask = self._ignore_stable_words(targets, graded, binary)

        return dict(targets=targets, graded=graded, binary=binary,
                    mask_stable=mask_stable, mask_changed=mask_changed, mask=mask)

    def _set_split(self, binary:np.array, mask:np.array, n_combinations: int = 100, percent: int = 40) -> tuple:
        """
        Args:
            binary(np.array): binary scores for target words
            mask(np.array): mask of changed/stable words to consider
            n_combinations(int, default=100): number of split for cross validation
            percent(int, default=40): percent of changed/stable word for each split
        Returns:
            train and test sets
        """
        set_seed(SEED)

        # -- Train and test sets --
        n_words = binary[mask].shape[0]
        n_words_per_test_set = int(n_words * percent / 100)

        idx_words = list(range(0, n_words))
        test_sets, train_sets = list(), list()
        for i in range(n_combinations):
            random.shuffle(idx_words)
            test_sets.append(idx_words[:n_words_per_test_set])
            train_sets.append(idx_words[n_words_per_test_set:])

        return train_sets, test_sets

    def _ignore_stable_words(self, targets: np.array, graded: np.array, binary: np.array) -> tuple:
        """Remove remove n stable words larger than the smallest changed words

        Args:
            targets(np.array): target words
            graded(np.array): graded score for each word
            binary(np.array): binary score for each word

        Returns:
            targets, graded, binary, mask_stable, mask_changed
        """

        # -- Indexes for changed and stable words --
        mask_changed = [i for i, k in enumerate(binary) if k]
        mask_stable = [i for i, k in enumerate(binary) if not k]

        if self.ignore_n_stable == 0:
            return targets, graded, binary, mask_stable, mask_changed, list(range(0, binary.shape[0]))
        
        # Min changed word
        min_changed = graded[mask_changed].min()

        # Top-graded stable words
        idx_max_stable = np.argpartition(graded[mask_stable], -self.ignore_n_stable)[-self.ignore_n_stable:]

        # Misleading values to ignore
        values_to_ignore = [graded[mask_stable][i] for i in idx_max_stable if graded[mask_stable][i] >= min_changed]

        # -- Mask --
        mask_stable = [i for i in mask_stable if all([graded[i] < v for v in values_to_ignore])]
        mask = [i for i, _ in enumerate(binary) if i in mask_stable or i in mask_changed]
        # remove high-graded stable words
        targets, graded, binary = targets[mask], graded[mask], binary[mask]

        mask_changed = [i for i, k in enumerate(binary) if k]
        mask_stable = [i for i, k in enumerate(binary) if not k]

        return targets, graded, binary, mask_stable, mask_changed, mask

    def _cross_validation(self, y_changed: np.array, y_stable: np.array,
                         train_sets: list, test_sets: list) -> dict:
        """
        Computes cross-validation statistics for a prediction task.

        Args:
            y_changed(np.array): The predicted values for changed words.
            y_stable(np.array): The predicted values for stable words.
            train_sets(list): A list of indices defining the samples in each training set.
            test_sets(list): A list of indices defining the samples in each test set.
        Returns:
            dict containing cross-validation stats
        """

        # -- Gold scores --
        y_true_stable = self.gt['graded'][self.gt['mask_stable']]
        y_true_changed = self.gt['graded'][self.gt['mask_changed']]
        labels_true_stable = np.zeros_like(y_true_stable)

        # -- Train and Test sets --
        cv_results = defaultdict(list)
        for i in range(0, len(train_sets)):
            # Corr (train, true)
            corr, pvalue = spearman(y_true_changed[train_sets[i]], y_changed[train_sets[i]])
            cv_results['cv_corr_train'].append(corr)
            cv_results['cv_pvalue_train'].append(pvalue)

            # Corr (test, true)
            corr, pvalue = spearman(y_true_changed[test_sets[i]], y_changed[test_sets[i]])
            cv_results['cv_corr_test'].append(corr)
            cv_results['cv_pvalue_test'].append(pvalue)

            # stable+test
            y = np.concatenate([y_stable, y_changed[test_sets[i]]])
            labels_true = np.concatenate([labels_true_stable, np.ones_like(y_changed[test_sets[i]])])

            # Corr (stable+test, true)
            corr, pvalue = spearman(np.concatenate([y_true_stable, y_true_changed[test_sets[i]]]), y)
            cv_results['cv_corr'].append(corr)
            cv_results['cv_pvalue'].append(pvalue)

            # Acc, Pr, Re, F1, AUC (stable+test, true)
            acc, thr = accuracy(labels_true, y)
            labels = np.array([1 if i >= thr else 0 for i in y])
            precision, recall, thresholds = precision_recall_curve(labels_true, labels)
            f1 = f1_score(labels_true, labels)

            if np.unique(labels).shape[0] != 2:
                auc_roc, auc_pr = 0, 0
            else:
                auc_roc = roc_auc_score(labels_true, labels)
                auc_pr = auc_score(recall, precision)

            cv_results['cv_precision'].append(precision)
            cv_results['cv_recall'].append(recall)
            cv_results['cv_f1'].append(f1)
            cv_results['cv_acc'].append(acc)
            cv_results['cv_thr'].append(thr)
            cv_results['cv_auc_roc'].append(auc_roc)
            cv_results['cv_auc_pr'].append(auc_pr)

        for k in list(cv_results):
            if k in ['cv_precision', 'cv_recall']: continue
            cv_results[f'{k}_mean'] = np.array(cv_results[k]).mean()
            cv_results[f'{k}_std'] = np.array(cv_results[k]).std()
            cv_results[f'{k}_max'] = np.array(cv_results[k]).max()
            cv_results[f'{k}_min'] = np.array(cv_results[k]).min()

        return cv_results

    def _cross_validation_regression(self, y_changed: np.array,
                                    y_stable: np.array,
                                    train_sets: list, test_sets: list) -> dict:
        """
        Computes cross-validation statistics for trained regressors

        Args:
            y_changed(np.array): The predicted values for changed words.
            y_stable(np.array): The predicted values for stable words.
            train_sets(list): A list of indices defining the samples in each training set.
            test_sets(list): A list of indices defining the samples in each test set.
        Returns:
            dict containing cross-validation stats
        """

        # -- Gold scores --
        y_true_stable = self.gt['graded'][self.gt['mask_stable']]
        y_true_changed = self.gt['graded'][self.gt['mask_changed']]
        labels_true_stable = np.zeros_like(y_true_stable)

        # -- Train and Test sets --
        cv_results = defaultdict(list)
        for i in range(0, len(train_sets)):
            # Regressor trained only on a train set of target words
            reg = LinearRegression().fit(y_changed[train_sets[i]], y_true_changed[train_sets[i]])

            # Regressor's prediction for the train set
            y = reg.predict(y_changed[train_sets[i]])

            # Corr (train, true)
            corr, pvalue = spearman(y_true_changed[train_sets[i]], y)
            cv_results['cv_corr_train'].append(corr)
            cv_results['cv_pvalue_train'].append(pvalue)

            # Corr (test, true)
            y = reg.predict(y_changed[test_sets[i]])
            corr, pvalue = spearman(y, y_true_changed[test_sets[i]])
            cv_results['cv_corr_test'].append(corr)
            cv_results['cv_pvalue_test'].append(pvalue)

            # Scores stable+test
            y = np.concatenate([y_stable, y_changed[test_sets[i]]])
            y = reg.predict(y)
            labels_true = np.concatenate([labels_true_stable, np.ones_like(y[labels_true_stable.shape[0]:])])

            # Corr (stable+test, true)
            corr, pvalue = spearman(np.concatenate([y_true_stable, y_true_changed[test_sets[i]]]), y)
            cv_results['cv_corr'].append(corr)
            cv_results['cv_pvalue'].append(pvalue)

            # Acc, Pr, Re, F1, AUC (stable+test, true)
            acc, thr = accuracy(labels_true, y)
            labels = np.array([1 if i >= thr else 0 for i in y])
            precision, recall, thresholds = precision_recall_curve(labels_true, labels)
            f1 = f1_score(labels_true, labels)

            if np.unique(labels).shape[0] != 2:
                auc_roc, auc_pr = 0, 0
            else:
                auc_roc = roc_auc_score(labels_true, labels)
                auc_pr = auc_score(recall, precision)

            cv_results['cv_precision'].append(precision)
            cv_results['cv_recall'].append(recall)
            cv_results['cv_f1'].append(f1)
            cv_results['cv_acc'].append(acc)
            cv_results['cv_thr'].append(thr)
            cv_results['cv_auc_roc'].append(auc_roc)
            cv_results['cv_auc_pr'].append(auc_pr)

        for k in list(cv_results):
            if k in ['cv_precision', 'cv_recall']: continue
            cv_results[f'{k}_mean'] = np.array(cv_results[k]).mean()
            cv_results[f'{k}_std'] = np.array(cv_results[k]).std()
            cv_results[f'{k}_max'] = np.array(cv_results[k]).max()
            cv_results[f'{k}_min'] = np.array(cv_results[k]).min()

        return cv_results
    
    def mix_measures(self, depth: int = 3, standardize: bool = True) -> pd.DataFrame:
        """
        Mix measure scores and evaluate performance for LSC detection

        Args:
            depth (int, default=3): the size of the possible combinations
            standardize (bool, default=True): If True, score are standardized

        Returns:
            pd.DataFrame containing the analysis results
        """

        # -- Prototype embeddings --
        PE1, PE2 = self.dh.load_all_prototype_embeddings(self.dataset, self.model, self.layers, self.name)

        # -- Distance matrix between embeddings --
        proto_dist = prototype_distance_matrix(PE1, PE2)

        # -- Prdedictions with different measures --
        y_preds_dict = dict()

        # Prototype Hausdorff Distance
        y_preds_dict['ProtoHD'] = lsc_measuring(PE1, PE2,
                                                lambda x1, x2: directed_hausdorff(x1, x2)[0])[self.gt['mask']]

        # Frobenius Norm
        y_preds_dict['Fnorm'] = lsc_measuring(proto_dist, None,
                                              lambda x: np.linalg.norm(x, 'fro'))[self.gt['mask']]

        # Condition Number
        y_preds_dict['Cond'] = lsc_measuring(proto_dist, None,
                                             lambda x: np.linalg.cond(x, 'fro'))[self.gt['mask']]

        # ERank
        y_preds_dict['Erank'] = lsc_measuring(proto_dist, None,
                                              lambda x: np.exp(entropy(PCA().fit(x).singular_values_)))[self.gt['mask']]

        # Average of PRT over different layers
        y_preds_dict['AvgPRT'] = lsc_measuring(proto_dist, None,
                                               lambda x: np.diag(x).mean())[self.gt['mask']]

        # set seed in order to make result reproducibile
        set_seed(SEED)

        for layer in range(1, self.layers + 1):
            y = self.dh.load_scores(self.dataset, self.model if 'Russian' not in self.dataset else self.model + '_40',
                                    layer)

            # old measures
            y_preds_dict[f'APD{layer}'] = y[y['measure'] == 'apd_cosine'].score.values[self.gt['mask']]
            y_preds_dict[f'PRT{layer}'] = y[y['measure'] == 'prt'].score.values[self.gt['mask']]
            y_preds_dict[f'HD{layer}'] = y[y['measure'] == 'hd'].score.values[self.gt['mask']]

            # new measures with different distance matrix
            # E1, E2 = self.dh.load_embeddings(self.dataset, self.model, layer, self.name)

            # random pick words
            # E1 = {word: E1[word][torch.randint(E1[word].shape[0], (100,))] for word in E1}
            # E2 = {word: E2[word][torch.randint(E2[word].shape[0], (100,))] for word in E2}

            # distance_matrix = {word: cdist(E1[word], E2[word], metric='cosine') for word in E1}

            # y_preds_dict[f'Fnorm{layer}'] = lsc_measuring(distance_matrix, None, lambda x: np.linalg.norm(x, 'fro'))[mask]
            # y_preds_dict[f'Erank{layer}'] = lsc_measuring(distance_matrix, None,
            #                                              lambda x: np.exp(entropy(PCA().fit(x).singular_values_)))[mask]

        # additional info
        y_preds_dict['word'] = self.gt['targets']
        y_preds_dict['change'] = self.gt['binary']

        # measure to combine
        measures_list = set([i for i in y_preds_dict.keys() if i not in ['word', 'change']])

        # -- Standardization --
        if standardize:
            for m in measures_list:
                y_preds_dict[m] = (y_preds_dict[m] - y_preds_dict[m].mean()) / y_preds_dict[m].std()

        # -- Combinations --
        combs = list()
        for i in range(2, depth + 1):
            combs += list(combinations(measures_list, i))

        # -- Result wrapper --
        for comb in combs:
            # - Geometrical mean
            comb_name = 'gmean ' + '-'.join(comb)
            values = gmean([y_preds_dict[m] for m in comb], axis=0)
            if not np.isnan(values).any():
                y_preds_dict[comb_name] = values

            # - Aritmetic mean
            comb_name = 'mean ' + '-'.join(comb)
            y_preds_dict[comb_name] = np.mean([y_preds_dict[m] for m in comb], axis=0)

            # - Sum
            comb_name = 'sum ' + '-'.join(comb)
            y_preds_dict[comb_name] = np.sum([y_preds_dict[m] for m in comb], axis=0)

            # - Linear regression
            comb_name = 'reg ' + '-'.join(comb)
            X = np.stack([y_preds_dict[m] for m in comb]).T
            reg = LinearRegression().fit(X, self.gt['graded'])
            y_preds_dict[comb_name] = reg.predict(X)
            y_preds_dict[comb_name + ' X'] = X

            # print(reg.score(X, y))
            # print(reg.coef_)
            # print(reg.intercept_)

        # -- Result wrapper --
        results = list()
        for comb_name in tqdm([k for k in y_preds_dict if not k.endswith(' X') and k not in ['word', 'change']], desc='Combining measures'):
            
            # score
            y = y_preds_dict[comb_name]
            
            # Corr, Acc, Pr, Re, F1, AUCs (preiction, true)
            corr, pvalue = spearman(self.gt['graded'], y)
            acc, thr = accuracy(self.gt['binary'], y)
            labels = np.array([1 if i >= thr else 0 for i in y])
            precision, recall, thresholds = precision_recall_curve(self.gt['binary'], labels)
            f1 = f1_score(self.gt['binary'], labels)

            if np.unique(labels).shape[0] != 2:
                auc_roc, auc_pr = 0, 0
            else:
                auc_roc = roc_auc_score(self.gt['binary'], labels)
                auc_pr = auc_score(recall, precision)

            # changed evaluation - Corr (changed, true)
            changed_corr, changed_pvalue = spearman(y[self.gt['mask_changed']],
                                                    self.gt['graded'][self.gt['mask_changed']])

            # stable evaluation - Corr (stable, true)
            stable_corr, stable_pvalue = spearman(y[self.gt['mask_stable']], self.gt['graded'][self.gt['mask_stable']])

            tmp = dict(idx=comb_name,
                       corr=corr, pvalue=pvalue,
                       acc=acc, thr=thr, auc_roc=auc_roc, auc_pr=auc_pr,
                       changed_corr=changed_corr, changed_pvalue=changed_pvalue,
                       stable_corr=stable_corr, stable_pvalue=stable_pvalue, precision=precision,
                       recall=recall, f1=f1,
                       y_preds=y)

            # cross validation
            if not comb_name.startswith('reg '):
                cv_results = self._cross_validation(y[self.gt['mask_changed']], y[self.gt['mask_stable']],
                                                   self.changed_train_sets, self.changed_test_sets)
            else:
                y = y_preds_dict[comb_name + ' X']
                cv_results = self._cross_validation_regression(y[self.gt['mask_changed']], y[self.gt['mask_stable']],
                                                              self.changed_train_sets, self.changed_test_sets)
            # cross validation info
            for k in cv_results:
                tmp[k] = cv_results[k]

            results.append(tmp)

        return pd.DataFrame(results)

    def mix_layers(self, depth: int = 4) -> pd.DataFrame:
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

        # -- Combinations --
        combs = list()
        for i in range(2, depth + 1):
            combs += list(combinations(list(range(1, self.layers + 1)), i))

        # -- Embeddings from time step 1 and 2 --
        E1, E2 = self.dh.load_all_embeddings(self.dataset, self.model, self.layers, self.name)

        # -- Results wrapper --
        results = list()
        for comb in tqdm(combs, desc='Combining embeddings'):
            for agg in ['mean', 'concat']:
                comb_name = '-'.join([f'L{l}' for l in comb])

                # self.dh.load_mix_embeddings(self.dataset, self.model, comb, agg)
                E1_mix, E2_mix = self.dh.mix_embeddings(E1, E2, comb, agg)

                # all apd experiments are on the same set
                set_seed(SEED)

                for k, v in {'apd': apd, 'prt': prt}.items():
                    y = lsc_measuring(E1_mix, E2_mix, v)[self.gt['mask']]

                    # Corr, Acc, F1, Pr, Re, AUC (prediction, true)
                    corr, pvalue = spearman(self.gt['graded'], y)
                    acc, thr = accuracy(self.gt['binary'], y)
                    labels = np.array([1 if i >= thr else 0 for i in y])

                    precision, recall, thresholds = precision_recall_curve(self.gt['binary'], labels)
                    f1 = f1_score(self.gt['binary'], labels)

                    if np.unique(labels).shape[0] != 2:
                        auc_roc, auc_pr = 0, 0
                    else:
                        auc_roc = roc_auc_score(self.gt['binary'], labels)
                        auc_pr = auc_score(recall, precision)

                    # changed evaluation - Corr (changed, true)
                    changed_corr, changed_pvalue = spearman(y[self.gt['mask_changed']],
                                                            self.gt['graded'][self.gt['mask_changed']])

                    # stable evaluation - Corr (stable, true)
                    stable_corr, stable_pvalue = spearman(y[self.gt['mask_stable']],
                                                          self.gt['graded'][self.gt['mask_stable']])

                    # permutation test
                    # pt_pvalue = permutation_test(graded, row['y_preds'])
                    # changed_pt_pvalue = permutation_test(graded[mask_changed], row['y_preds'][mask_changed])
                    # stable_pt_pvalue = permutation_test(graded[mask_stable], row['y_preds'][mask_stable])

                    tmp = dict(idx=comb_name, agg=agg,
                               corr=corr, pvalue=pvalue,
                               acc=acc, thr=thr,
                               auc_roc=auc_roc, auc_pr=auc_pr,
                               changed_corr=changed_corr, changed_pvalue=changed_pvalue,
                               stable_corr=stable_corr, stable_pvalue=stable_pvalue,
                               score=y,
                               measure=k, precision=precision, recall=recall, f1=f1)

                    # cross validation test
                    cv_results = self._cross_validation(y[self.gt['mask_changed']], y[self.gt['mask_stable']],
                                                       self.changed_train_sets, self.changed_test_sets)
                    for k in cv_results:
                        tmp[k] = cv_results[k]

                    results.append(tmp)
        
        return pd.DataFrame(results)

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
    parser.add_argument('-s', '--stable_words_to_ignore',
                        type=int, default=0,
                        help='Remove remove n stable words larger than the smallest changed words')
    args = parser.parse_args()

    # output folder
    out_folder = f'{args.output_folder}/LSC/{args.dataset}/{args.model}'
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    b = BruteForce(args.main_folder, args.dataset, args.model, args.layers,
                   ignore_n_stable=args.stable_words_to_ignore)

    res = b.mix_measures(args.depth, standardize=True)
    res.to_csv(f'{out_folder}/mix_std_measures_{args.stable_words_to_ignore}.txt', index=False, sep='\t')

    res = b.mix_measures(args.depth, standardize=False)
    res.to_csv(f'{out_folder}/mix_measures_{args.stable_words_to_ignore}.txt', index=False, sep='\t')

    res = b.mix_layers(args.depth + 1)
    res.to_csv(f'{out_folder}/mix_layers_{args.stable_words_to_ignore}.txt', index=False, sep='\t')
