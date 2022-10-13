
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', required=True)
    parser.add_argument('--input-files', nargs='+', required=True)
    parser.add_argument('--lhs', default='left')
    parser.add_argument('--target', default='hit')
    parser.add_argument('--rhs', default='right')
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
    model = AutoModel.from_pretrained(args.modelname)
    model.eval()

    for path in args.input_files:
        data = pd.read_csv(path)
        for heading in [args.lhs, args.target, args.rhs]:
            data[heading] = data[heading].transform(normalise)
        sents = data[[args.lhs, args.target, args.rhs]].agg(' '.join, axis=1).values.tolist()
        starts = data[args.lhs].str.len() + 1
        stops = data[args.lhs].str.len() + 1 + data[args.target].str.len()
        sents, spans = encode_data(tokenizer, sents, starts, stops, sym=None)

        # Generate sentence embeddings
        with torch.no_grad():
            ids = tokenizer(sents, return_tensors='pt', padding=True).to(device)
            output = model(**ids)

        embeddings = []
        for idx, (start, stop) in enumerate(spans):
            embeddings.append(output['last_hidden_state'][idx][start:stop].mean(0).cpu().numpy())

        data['embedding'] = embeddings
        prefix, suffix = os.path.splitext(path)
        data.to_parquet(''.join([prefix, '.embeddings.parquet']))
