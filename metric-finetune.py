
import collections
import os
import itertools
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModel, AutoTokenizer

from preprocess import encode_data


def get_metrics(trues, preds):
    return {'accuracy': metrics.accuracy_score(trues, preds),
            'f1-micro': metrics.f1_score(trues, preds, average='micro'),
            'f1-macro': metrics.f1_score(trues, preds, average='macro')}


def sample_up_to_n(g, n):
    if len(g) <= n:
        return g
    return g.sample(n=n)


def get_dataset(sents, spans, labels):
    dataset = {'text': sents, 'spans': spans}
    dataset['label'] = labels
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_encodings(model, tokenizer, batch_data, max_batch_size=20, device='cpu'):
    output = []
    inputs = tokenizer(batch_data['text'], return_tensors='pt', padding=True)
    total = len(inputs['input_ids'])
    for b in range(0, total, max_batch_size):
        b_inputs = {key: val[b:min(b + max_batch_size, total)] for key, val in inputs.items()}
        b_inputs = {key: val.to(device) for key, val in b_inputs.items()}
        for idx, hidden in enumerate(model(**b_inputs)['last_hidden_state']):
            start, end = batch_data['spans'][idx + b]
            output.append(hidden[start:end].mean(dim=0))

    return torch.stack(output)


def get_centroids(model, tokenizer, support_data, **kwargs):
    output = get_encodings(model, tokenizer, support_data, **kwargs)
    return get_centroids_(output, support_data)


def get_centroids_(output, support_data):
    last = 0
    centroids = []
    for _, g in (itertools.groupby(support_data['label'])):
        g = list(g)
        centroids.append(output[last: last + len(g)].mean(0))
        last += len(g)
    return torch.stack(centroids)


def train_epoch(
        model, tokenizer, train_dataset, optimizer,
        device='cpu', batch_size=20, max_batch_size=20, max_support=20):

    index = np.arange(len(train_dataset))
    label = np.array(train_dataset['label'])
    perm = np.random.permutation(index)

    tloss = 0
    for b_id, batch in enumerate(range(0, len(perm), batch_size)):
        query_ids = perm[batch: batch+batch_size]
        query_data = train_dataset[query_ids]
        query = get_encodings(model, tokenizer, query_data,
            max_batch_size=max_batch_size, device=device)

        support_ids = np.delete(index, query_ids)
        # support ids, downsample to max support
        sample = pd.DataFrame(
            {'index': support_ids, 'label': label[support_ids]}
        ).groupby("label").apply(lambda c: sample_up_to_n(c, max_support))
        support_ids_sample = sample['index'].values
        support_data = train_dataset[support_ids_sample]
        centroids = get_centroids(model, tokenizer, support_data,
            max_batch_size=max_batch_size, device=device)
        scores = query @ centroids.t()
        targets = torch.tensor(query_data['label']).to(device)
        loss = F.cross_entropy(scores, targets)
        loss.backward()
        tloss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if b_id % 10 == 0:
            print("Batch {}: {:g}".format(b_id, tloss / 10))
            tloss = 0


# run evaluation
def predict(model, tokenizer, train_dataset, dataset,
            device='cpu', max_batch_size=20, max_support=20):

    index = np.arange(len(train_dataset))
    sample = pd.DataFrame(
        {'index': index, 'label': np.array(train_dataset['label'])}
    ).groupby("label").apply(lambda c: sample_up_to_n(c, max_support))

    support_data = train_dataset[sample['index'].values]
    with torch.no_grad():
        centroids = get_centroids(model, tokenizer, support_data,
            device=device, max_batch_size=max_batch_size)
        query = get_encodings(model, tokenizer, dataset,
            device=device, max_batch_size=max_batch_size)
        scores = query @ centroids.t()

    preds = scores.argmax(dim=1).cpu().numpy()
    scores = np.max(softmax(scores.detach().cpu().numpy(), axis=1), axis=1)

    return preds, scores


def predict_multiple_samples(
        model, tokenizer, train_dataset, dataset,
        device='cpu', max_batch_size=20, max_support=20, n_preds=20):

    with torch.no_grad():
        output = get_encodings(model, tokenizer, train_dataset,
            device=device, max_batch_size=max_batch_size)
        query = get_encodings(model, tokenizer, dataset,
            device=device, max_batch_size=max_batch_size)

    samples = []
    index = np.arange(len(train_dataset))
    for _ in range(n_preds):
        sample = pd.DataFrame(
            {'index': index, 'label': np.array(train_dataset['label'])}
        ).groupby("label").apply(lambda c: sample_up_to_n(c, max_support))
        centroids = get_centroids_(
            output[sample['index'].values],
            train_dataset[sample['index'].values])
        scores = query @ centroids.t()
        samples.append(scores.argmax(dim=1).cpu().numpy())

    samples = np.stack(samples)
    preds, scores = [], []
    for ix in range(len(dataset)):
        values, counts = np.unique(samples[:, ix], return_counts=True)
        pred = values[np.argmax(counts)]
        preds.append(pred)
        scores.append(counts[np.argmax(counts)] / counts.sum())

    return preds, scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', required=True)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--lhs', default='left')
    parser.add_argument('--target', default='hit')
    parser.add_argument('--rhs', default='right')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-batch-size', type=int, default=15)
    parser.add_argument('--max-support', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=10)
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
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})

    data = pd.read_csv(args.input_file)
    # data = pd.read_csv('../geur/reuk_300.csv')
    # prepare sents
    for heading in [args.lhs, args.target, args.rhs]:
        data[heading] = data[heading].transform(normalise)
    sents = data[[args.lhs, args.target, args.rhs]].agg(' '.join, axis=1).values.tolist()
    starts = data[args.lhs].str.len() + 1
    stops = data[args.lhs].str.len() + 1 + data[args.target].str.len()
    sents, spans = encode_data(tokenizer, sents, starts, stops)
    sents, spans = np.array(sents), np.array(spans)
    # prepare labels
    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label]])

    cv = StratifiedKFold(10, shuffle=True, random_state=135)
    folds = []

    for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
        hapaxes = set([l for l, c in collections.Counter(y[train]).items() if c == 1])
        train, dev = next(StratifiedKFold(10, shuffle=True).split(
            # np.zeros(len(train)), y[train]))
            [i for i in train if y[i] not in hapaxes],
            [y[i] for i in train if y[i] not in hapaxes]))

        train_dataset = get_dataset(sents[train], spans[train], y[train])
        dev_dataset = get_dataset(sents[dev], spans[dev], y[dev])
        test_dataset = get_dataset(sents[test], spans[test], y[test])

        model = AutoModel.from_pretrained(args.modelname).to(device)
        model.resize_token_embeddings(len(tokenizer))
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)

        for epoch in range(args.epochs):
            model.train()
            train_epoch(model, tokenizer, train_dataset, optimizer,
                device=device, max_batch_size=args.max_batch_size, batch_size=args.batch_size,
                max_support=args.max_support)
            model.eval()
            print("Epoch {}".format(epoch + 1))
            preds, _ = predict(model, tokenizer, train_dataset, dev_dataset,
                device=device, max_batch_size=args.max_batch_size)
            print(get_metrics(dev_dataset['label'], preds))
            print("Epoch {}".format(epoch + 1))
            preds, _ = predict(model, tokenizer, train_dataset, dev_dataset, max_support=50,
                device=device, max_batch_size=args.max_batch_size)
            print(get_metrics(dev_dataset['label'], preds))
            print("Epoch {}".format(epoch + 1))
            preds, _ = predict_multiple_samples(model, tokenizer, train_dataset, dev_dataset,
                device=device, max_batch_size=args.max_batch_size)
            print(get_metrics(dev_dataset['label'], preds))

        preds, scores = predict(model, tokenizer, train_dataset, test_dataset,
            device=device, max_batch_size=args.max_batch_size, max_support=50)

        folds.append(pd.DataFrame({'fold': fold, 'test': test, 'scores': scores, 'preds': preds}))

    prefix, suffix = os.path.splitext(args.input_file)
    pd.concat(folds).to_parquet(''.join([prefix, '.metric-finetune.results.parquet']))
