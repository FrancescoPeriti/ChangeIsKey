import os
import json
import torch
import random
import numpy as np
from torch import nn
from pathlib import Path
from datasets import logging as dataset_logging
from transformers import logging as transformers_logging
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

dataset_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

SEED = 42


def set_seed(seed: int) -> None:
    """
    This function sets the seed for the random number generators in Python's built-in random module, NumPy,
    PyTorch CPU, and PyTorch GPU. This is useful for ensuring reproducibility of results.

    Args:
        seed (int): The seed to set the random number generators to.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BertFineTuning:
    """A class to fine-tune a pre-trained BERT model on MLM."""

    def __init__(self, pretrained: str = 'bert-base-uncased'):
        """
        Initializes a pre-trained BERT model.

        Args:
            pretrained (str, default='bert-base-uncased'): The pre-trained model name or path to load.
        """

        # - set seed and device
        self._device = self._set_seed_and_device()

        # - load Hugginface tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained)

        # - if multiple GPUs are available
        #if torch.cuda.device_count() > 1:
        #    self.model = torch.nn.DataParallel(self.model)

        # - initialize model for training
        _ = self.model.to(self._device)
        _ = self.model.train()

        self.pretrained = pretrained
        
    def _set_seed_and_device(self) -> object:
        """
        Determines the device (CPU or GPU) to run the model on.

        Returns:
            object, a torch device object.
        """

        # - set seed
        set_seed(SEED)

        # - set device (GPU if available)
        self._device_name = "cuda" if torch.cuda.is_available() else "cpu"

        return torch.device(self._device_name)

    def _tokenize_factory(self, tokenizer: object, max_length: int = None) -> object:
        """
        A factory function that returns a function used to tokenize text using the provided tokenizer.

        Args:
            tokenizer(object): The tokenizer object used for tokenization
            max_length(int, optional): The maximum sequence length to use. If None, the max_length possibile is used.

        Returns:
            function
        """
        max_length = max_length if max_length is not None else tokenizer.model_max_length

        def tokenize(examples) -> dict:
            """Tokenization function"""
            return tokenizer(examples["sent"], return_tensors='pt',
                             padding="max_length", max_length=max_length, truncation=True).to(self._device)

        return tokenize
    
    def _load_dataset(self, dataset: list) -> Dataset:
        return load_dataset('text', data_files=dataset, split='train').rename_column('text', 'sent')

    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        """
        Tokenizes the text data in a dataset using the BERT tokenizer.

        Args:
            dataset (Dataset): A Dataset object containing the text data to be tokenized.
            max_length (int): The maximum length of the sequences after tokenization.

        Returns:
            A Dataset object containing the tokenized text data.
        """

        # - create function
        tokenize_func = self._tokenize_factory(self.tokenizer, max_length)

        # - tokenize dataset
        dataset = dataset.map(tokenize_func, batched=True)

        # - set format to torch
        dataset.set_format('torch')

        return dataset

    def train(self, dataset: list, max_length: int = 512,
              data_collator_args: dict = None,
              training_argument_args: dict = None):

        """
        Fine-tune a pre-trained language model.

        Args:
            dataset(list): List of paths in which the dataset is split.
            max_length(int, default=512): The maximum length of the input sequences.
            data_collator_args(dict, default=None): Arguments to pass to the data collator.
            training_argument_args(dict, default=None): Arguments to pass to the training arguments.

        Returns:
            None
        """

        dataset = self._load_dataset(dataset) 
        
        # tokenization
        tokenized_datasets = self._tokenize_dataset(dataset, max_length)

        # arguments
        if data_collator_args is None:
            data_collator_args = dict()

        if training_argument_args is None:
            training_argument_args = dict(output_dir=self.pretrained)
        if 'output_dir' not in training_argument_args:
            training_argument_args.update({'output_dir': self.pretrained})

        # training
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        **data_collator_args)
        training_args = TrainingArguments(**training_argument_args)
        trainer = Trainer(self.model, training_args, train_dataset=tokenized_datasets,
                          data_collator=data_collator, tokenizer=self.tokenizer)
        trainer.train()

        # store model
        folder = os.path.dirname(os.path.realpath(training_argument_args['output_dir']))
        Path(folder).mkdir(parents=True, exist_ok=True)
        trainer.save_model(training_argument_args['output_dir'])


class BertFineTuningFull(BertFineTuning):
    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        return super()._tokenize_dataset(dataset, max_length).rename_column('sent', 'text')

class BertFineTuningFilter(BertFineTuning):
    def _load_dataset(self, dataset: list) -> Dataset:
        """
        This method loads a dataset from a file and returns it as a Dataset object.

        Args:
            dataset (str): A string representing the path to the dataset file.

        Returns:
            A Dataset object representing the loaded dataset.
        """

        rows = list()
        for filename in dataset:
            with open(filename, mode='r', encoding='utf-8') as f:
                rows += [json.loads(line) for line in f]
        return Dataset.from_list(rows)

    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        return super()._tokenize_dataset(dataset, max_length).remove_columns(['start', 'end', 'lemma', 'token', 'sentidx'])

class BertFineTuningSeparate(BertFineTuning):
    def __init__(pretrained: str = 'bert-base-uncased', tokens: list=None):
        super().__init__(pretrained)
        self.tokens = tokens

    def _reset_weights(self, tokens: list):

        for token in tokens:
            token_ids = self.tokenizer.convert_tokens_to_ids([token])
            weights = self.model.bert.embeddings.word_embeddings.weight.data

            if len(token_ids) == 1:
                weights[token_ids[0]]=torch.rand(768)
                new_emb=nn.Embedding.from_pretrained(weights, padding_idx=0, freeze=False)
                
            else:
                self.tokenizer.add_tokens(token)
                new_weights = torch.cat((weights, torch.rand(768)), 0)
                new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)

            self.model.bert.embeddings.word_embeddings = new_emb

    def _load_dataset(dataset: list) -> Dataset:
        dataset = super()._load_dataset(dataset)

        if self.tokens is not None:
            self._reset_weights(self.tokens)

        return dataset


if __name__ == '__main__':
    import argparse
    import importlib
    from pathlib import Path

    parser = argparse.ArgumentParser(prog='BertFineTuning',
                                     description='Fine-tuning BERT for each target word')
    parser.add_argument('-t', '--tokenized_dataset',
                        type=str,
                        help='A string representing the directory path to a tokenized dataset for LSC detection. '
                             'This dataset should contain pre-tokenized text for the target words.')
    parser.add_argument('-m', '--model',
                        type=str,
                        help='A string representing the name of the Hugging Face pre-trained model to use for embedding extraction.')
    parser.add_argument('-M', '--max_length',
                        type=int, default=512,
                        help='An integer representing the maximum sequence length to use for the embedding extraction process. '
                             'Default value is 512.')
    parser.add_argument('-e', '--epochs',
                        type=int, default=4,
                        help='An integer representing the number of encoder layers of the pre-trained model to use for embedding extraction. '
                             'Default value is 12.')
    parser.add_argument('-b', '--batch_size',
                        type=int, default=64,
                        help='An integer representing the batch size to use for the embedding extraction process. '
                             'Default value is 8.')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Dirname where embeddings will be stored')
    parser.add_argument('-T', '--targets', type=str, default=None, 
                        help='A string representing the directory path to a text file containing the target words.')
    parser.add_argument('-k', '--finetune_class', type=str, default='BertFineTuningFull',
                        help='Class to instanziate.')
    args = parser.parse_args()

    # reflection -> get the class to instanziate
    module = importlib.import_module(__name__)
    finetuning_class = getattr(module, args.finetune_class)

    # Finetune BERT
    if args.targets is not None and args.finetune_class == 'BertFineTuningSeparate':
        targets=[w.strip() for w in open(args.targets, mode='r', encoding='utf-8').readlines()]
        bert = finetuning_class(args.model, targets)
    else:
        bert = finetuning_class(args.model)

    if args.finetune_class == 'BertFineTuningSeparate':
        output1=output+'-corpus1'
        output2=output+'-corpus2'

        training_argument_args = dict(output_dir=output1, num_train_epochs=args.epochs,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size)
        bert.train([paths[0]], max_length=args.max_length, data_collator_args=None, 
            training_argument_args=training_argument_args, tokens=targets)


        training_argument_args = dict(output_dir=output2, num_train_epochs=args.epochs,
                                  per_device_train_batch_size=args.batch_size,
                                  per_device_eval_batch_size=args.batch_size)
        bert.train([paths[1]], max_length=args.max_length, data_collator_args=None, 
        training_argument_args=training_argument_args, tokens=targets)

    else:
        if args.mode == 'BertFineTuningFilter':
            # fine-tune over target sentences
            paths = [str(path_) for path_ in Path(f'{args.tokenized_dataset}/corpus1/token/').glob('*.txt')]
            paths += [str(path_) for path_ in Path(f'{args.tokenized_dataset}/corpus2/token/').glob('*.txt')]
            output=args.output
            
        elif args.mode == 'BertFineTuningFull':
            # fine-tune over the whole corpus
            paths=[f'{args.tokenized_dataset}/corpus1/token/corpus1.txt', f'{args.tokenized_dataset}/corpus2/token/corpus2.txt']
            output=args.output

        training_argument_args = dict(output_dir=output, num_train_epochs=args.epochs,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size)
        bert.train(paths, max_length=args.max_length, data_collator_args=None, training_argument_args=training_argument_args, 
            tokens=None)
    