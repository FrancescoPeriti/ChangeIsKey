import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from extraction import WordEmbeddingExtraction

parser = argparse.ArgumentParser(prog='Embedding extraction',
                                 description='Extract the embeddeddings for each target word')
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
parser.add_argument('-l', '--layers',
                    type=int, default=12,
                    help='An integer representing the number of encoder layers of the pre-trained model to use for embedding extraction. '
                         'Default value is 12.')
parser.add_argument('-b', '--batch_size',
                    type=int, default=8,
                    help='An integer representing the batch size to use for the embedding extraction process. '
                         'Default value is 8.')
parser.add_argument('-o', '--output',
                    type=str,
                    help='Dirname where embeddings will be stored')
args = parser.parse_args()


w = WordEmbeddingExtraction(args.model)

tokenization_input = f'{args.tokenized_dataset}.txt'
embeddings_output = f'{args.output}/{args.model.replace("/", "_")}/'

# extraction
embeddings = w.extract(dataset=tokenization_input, batch_size=args.batch_size,
                       max_length=args.max_length, agg_sub_words='mean',
                       layers=args.layers, sampling=0)

# store embeddings
for l in range(1, args.layers + 1):
    Path(f'{embeddings_output}').mkdir(parents=True, exist_ok=True)
    torch.save(embeddings[l].to('cpu'), f'{embeddings_output}/{l}.pt')

