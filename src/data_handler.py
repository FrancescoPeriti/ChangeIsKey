import torch
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict


class DataHandler:
    def __init__(self, data_folder, datasets_folder:str='datasets', scores_folder:str='scores',
                 embeddings_folder:str='contextualized_embeddings', attentions_folder:str='attentions'):
        self.data_folder = data_folder
        self.datasets_folder=datasets_folder
        self.scores_folder=scores_folder
        self.embeddings_folder=embeddings_folder
        self.attentions_folder=attentions_folder

    def load_targets(self, dataset: str, name=None) -> pd.DataFrame:
        name = 'targets' if name is None else f'targets_{name}'
        filename = f'{self.data_folder}/{self.datasets_folder}/LSC/{dataset}/{name}.txt'
        return pd.read_csv(filename, sep='\t', names=['word']).sort_values('word')

    def load_binary(self, dataset: str, name=None) -> pd.DataFrame:
        name = 'binary' if name is None else f'binary_{name}'
        filename = f'{self.data_folder}/{self.datasets_folder}/LSC/{dataset}/truth/{name}.txt'
        return pd.read_csv(filename, sep='\t', names=['word', 'score']).sort_values('word')

    def load_graded(self, dataset: str, name=None) -> pd.DataFrame:
        name = 'graded' if name is None else f'graded_{name}'
        filename = f'{self.data_folder}/{self.datasets_folder}/LSC/{dataset}/truth/{name}.txt'
        return pd.read_csv(filename, sep='\t', names=['word', 'score']).sort_values('word')

    def load_scores(self, dataset: str, model: str, layer: int = 12) -> pd.DataFrame:
        filename = f'{self.data_folder}/{self.scores_folder}/LSC/{dataset}/{model}/{layer}/token.txt'
        return pd.read_csv(filename, sep='\t', names=['word', 'measure', 'score']).sort_values('word')

    def load_embeddings(self, dataset: str, model: str, layer: int = 12, name=None) -> tuple:
        folder1 = f'{self.data_folder}/{self.embeddings_folder}/LSC/{dataset}/{model}/corpus1/token/{layer}'
        folder2 = f'{self.data_folder}/{self.embeddings_folder}/LSC/{dataset}/{model}/corpus2/token/{layer}'

        targets = self.load_targets(dataset, name).word.values

        Embs1, Embs2 = dict(), dict()
        for target in targets:
            filename1 = f'{folder1}/{target}.pt'
            filename2 = f'{folder2}/{target}.pt'
            Embs1[target] = torch.load(filename1)
            Embs2[target] = torch.load(filename2)

        return Embs1, Embs2

    def load_attentions(self, dataset: str, model: str, layer: int = 12, name=None) -> tuple:
        folder1 = f'{self.data_folder}/{self.attentions_folder}/LSC/{dataset}/{model}/corpus1/token/{layer}/'
        folder2 = f'{self.data_folder}/{self.attentions_folder}/LSC/{dataset}/{model}/corpus2/token/{layer}/'

        targets = self.load_targets(dataset, name).word.values

        Attn1, Attn2 = dict(), dict()
        for filename1, filename2 in zip(Path(folder1).glob('*.pickle'), Path(folder2).glob('*.pickle')):
            if filename1.stem not in targets: continue
            Attn1[filename1.stem] = pickle.load(open(filename1, mode='rb'))
            Attn2[filename2.stem] = pickle.load(open(filename2, mode='rb'))

        return Attn1, Attn2

    def load_mix_embeddings(self, dataset: str, model: str, layers: list, agg: str = 'concat', name:str=None):
        targets = self.load_targets(dataset).word.values

        E1_mix = defaultdict(list)
        E2_mix = defaultdict(list)

        for layer in layers:
            E1, E2 = self.load_embeddings(dataset, model, layer, name)

            for word in targets:
                E1_mix[word].append(E1[word])
                E2_mix[word].append(E2[word])

        for word in targets:
            if agg == 'mean':
                E1_mix[word] = torch.stack(E1_mix[word]).mean(axis=0)
                E2_mix[word] = torch.stack(E2_mix[word]).mean(axis=0)

            if agg == 'concat':
                E1_mix[word] = torch.hstack(E1_mix[word])
                E2_mix[word] = torch.hstack(E2_mix[word])

        return E1_mix, E2_mix

    def load_prototype_embeddings(self, dataset: str, model: str, layer: int = 12, name: str = None) -> tuple:
        E1, E2 = self.load_embeddings(dataset, model, layer, name)

        for word in E1.keys():
            E1[word] = E1[word].mean(axis=0)
            E2[word] = E2[word].mean(axis=0)

        return E1, E2

    def load_all_prototype_embeddings(self, dataset: str, model: str, layers: int = 12, name: str = None) -> tuple:
        E1, E2 = defaultdict(list), defaultdict(list)
        for layer in range(1, layers + 1):
            E1_layer, E2_layer = self.load_prototype_embeddings(dataset, model, layer, name)

            for word in E1_layer.keys():
                E1[word].append(E1_layer[word])
                E2[word].append(E2_layer[word])

        for word in E1.keys():
            E1[word] = torch.stack(E1[word])
            E2[word] = torch.stack(E2[word])

        return E1, E2


    def load_all_embeddings(self, dataset: str, model: str, layers: int = 12, name=None) -> tuple:
        E1, E2 = dict(), dict()
        for layer in range(1, layers+1):
            E1[layer], E2[layer] = self.load_embeddings(dataset, model, layer, name)

        return E1, E2
    
    def mix_embeddings(self, E1: dict, E2: dict, layers:list, agg:str):
        E1_mix = defaultdict(list)
        E2_mix = defaultdict(list)

        for layer in layers:
            for word in E1[layer]:
                E1_mix[word].append(E1[layer][word])
                E2_mix[word].append(E2[layer][word])

        for word in E1[layers[0]]:
            if agg == 'mean':
                E1_mix[word] = torch.stack(E1_mix[word]).mean(axis=0)
                E2_mix[word] = torch.stack(E2_mix[word]).mean(axis=0)

            if agg == 'concat':
                E1_mix[word] = torch.hstack(E1_mix[word])
                E2_mix[word] = torch.hstack(E2_mix[word])

        return E1_mix, E2_mix
