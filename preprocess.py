
"""
preprocess.py: This file provides two utility functions that 
help extracting contextual embeddings from a transformer model. 
Mostly dealing with subtokenization. For more information on how 
to run this script, run the command ‘python preprocess.py --help’ 
in the command line.
"""

__author__    = "Enrique Manjavacas"
__copyright__ = "Copyright 2022, Enrique Manjavacas"
__license__   = "GPL"
__version__   = "1.0.1"

import os
import collections

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Find start and end index of target word
def read_data(lhs, target, rhs, mask=None):
    """
    Computes the spans (start and end offsets) from a list of segmented sentences
    """
    sents, starts, ends = [], [], []
    for (lhs_i, target_i, rhs_i) in zip(lhs, target, rhs):
        if mask is not None:
            target_i = mask
        sents.append((lhs_i + ' ' + target_i + ' ' + rhs_i).strip())
        start = len(lhs_i) + 1 if lhs_i else 0
        starts.append(start)
        ends.append(start + len(target_i))
    return sents, starts, ends


def encode_data(tokenizer, sents, starts, ends, sym='[TGT]'):
    """
    Transform input sentences into tokenized input for the model.

    Input
    =====
    - tokenizer : transformers tokenizer
    - sents : list of strings
    - starts : list of ints pointing at the index at which the target word starts
    - ends : list of ints pointing at the index at which the target word ends
    - sym : string, symbol to use to signalize what the target word is (make sure
        you add it to the tokenizer vocabulary if not already there)

    Output
    ======
    - output_sents : list of strings
    - spans : list of tuples (start, end), where `start` is the index of
        the first subtoken corresponding to the target word, `end` the index
        of the last one.
    """
    output_sents, spans = [], []
    for sent, char_start, char_end in zip(sents, starts, ends):
        # insert target symbols
        if sym is not None:
            sent = sent[:char_start] + f'{sym} ' + sent[char_start:char_end] + f' {sym}' + sent[char_end:]
        output_sents.append(sent)

        sent = tokenizer.encode_plus(sent, return_offsets_mapping=True)
        # transform character indices to subtoken indices
        target_start = target_end = None
        if sym is not None:
            char_start += len(sym) + 1
            char_end += len(sym) + 1
        for idx, (token_start, token_end) in enumerate(sent['offset_mapping']):
            if token_start == token_end == 0: continue
            if token_start == char_start:
                target_start = idx
            if token_end == char_end:
                target_end = idx
        if target_start is None or target_end is None:
            raise ValueError
        spans.append((target_start, target_end + 1))

    return output_sents, spans


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
    "This script extracts embeddings for the target words of given input sentences "
    "and stores them into a parquet file, with the same name as the source file, but "
    "with a .parquet extension.")
    parser.add_argument('--modelname', required=True, 
                        help="The name of a transformer model (huggingface).")
    parser.add_argument('--input-files', nargs='+', required=True, 
                        help="CSV file with at least three field specifying the left "
                        "and right context as well as the target word.")
    parser.add_argument('--lhs', default='left', help='Name of the left context column.')
    parser.add_argument('--target', default='hit', help='Name of the left context column.')
    parser.add_argument('--rhs', default='right', help='Name of the left context column.')
    args = parser.parse_args()
    args = parser.parse_args()

    # Normalise whitespaces
    def normalise(example):
        return ' '.join(example.split())

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('PyTorch is using CUDA enabled GPU')
    else:
        device = torch.device('cpu')
        print('PyTorch is using CPU')

    tokenizer = AutoTokenizer.from_pretrained(args.modelname)
    model = AutoModel.from_pretrained(args.modelname).to(device)
    model.eval()

    for path in args.input_files:
        print(path)
        data = pd.read_csv(path)
        for heading in [args.lhs, args.target, args.rhs]:
            data[heading] = data[heading].transform(normalise)
        sents, starts, ends = read_data(data[args.lhs], data[args.target], data[args.rhs])
        sents, spans = encode_data(tokenizer, sents, starts, ends, sym=None)
        print(collections.Counter(
                [tuple(tokenizer.convert_ids_to_tokens(tokenizer.encode(s))[a:b])
                for s, (a, b) in zip(sents, spans)]))

        # Generate sentence embeddings
        with torch.no_grad():
            ids = tokenizer(sents, return_tensors='pt', padding=True).to(device)
            output = model(**ids)

        embeddings = []
        for idx, (start, stop) in enumerate(spans):
            embedding = output['last_hidden_state'][idx][start:stop].mean(0).cpu().numpy()
            if any(np.isnan(embedding)):
                raise
            embeddings.append(embedding)

        data['embedding'] = embeddings
        prefix, suffix = os.path.splitext(path)
        data.to_parquet(''.join([prefix, '.embeddings.parquet']))


