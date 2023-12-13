
"""
finetune-cv.py: This script performs training on the provided train file, and
inference on the provided test file. For more information on how to run this 
script, run the command ‘python finetune.py --help’ in the command line.
"""

__author__    = "Enrique Manjavacas"
__copyright__ = "Copyright 2022, Enrique Manjavacas"
__license__   = "GPL"
__version__   = "1.0.1"

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from preprocess import encode_data, read_data


def get_dataset(tokenizer, sents, spans, labels):
    dataset = {'text': sents, 'spans': spans}
    dataset['label'] = labels
    dataset = Dataset.from_dict(dataset)

    return dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True, max_length=512),
        batched=True
    ).remove_columns('text')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(       
        "This file performs finetuning on a given train file and a given pre-trained "
        "transformer model. "
        "Evaluation is done with Cross-Validation. After training the cross-validation "
        "results on each fold are stored to a finetune.results.parquet file."
        "If a test-file was provided, then the model trained in each iteration is used "
        "for prediction on the test file and the results are output to a file with the "
        "suffix: finetune.test-results.parquet")
    parser.add_argument('--modelname', required=True,
                        help="The name of a transformer model (huggingface).")
    parser.add_argument('--input-file', required=True, help="File to do training on.")
    parser.add_argument('--test-file', help="File to do inference on.")
    parser.add_argument('--infix', default='', help='String to add to the results file.')
    parser.add_argument('--label', required=True, help="Name of the label column.")
    parser.add_argument('--lhs', default='left', help='Name of the left context column.')
    parser.add_argument('--target', default='hit', help='Name of the left context column.')
    parser.add_argument('--rhs', default='right', help='Name of the left context column.')
    parser.add_argument('--epochs', type=int, default=6, help="Number of epochs to train.")
    parser.add_argument('--folds', default=10, type=int)
    parser.add_argument('--output-dir', required=True, help="Directory to store the finetuned model.")
    parser.add_argument('--mask-target', action='store_true')
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
    data = data.fillna('')
    for heading in [args.lhs, args.target, args.rhs]:
        data[heading] = data[heading].astype(str).transform(normalise)
    mask = None
    if args.mask_target:
        mask = tokenizer.mask_token
    sents, starts, ends = read_data(
        data[args.lhs], data[args.target], data[args.rhs], mask=mask)
    sents, spans = encode_data(tokenizer, sents, starts, ends)
    sents, spans = np.array(sents), np.array(spans)
    # prepare labels
    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label]])

    cv = StratifiedKFold(args.folds, shuffle=True, random_state=135)
    folds = []
    test_folds = []
    test_data = None
    if args.test_file:
        test_data = pd.read_csv(args.test_file)
        for heading in [args.lhs, args.target, args.rhs]:
            test_data[heading] = test_data[heading].astype(str).transform(normalise)
        test_sents, test_starts, test_stops = read_data(
            test_data[args.lhs], test_data[args.target], test_data[args.rhs], mask=mask)
        test_sents, test_spans = encode_data(tokenizer, test_sents, test_starts, test_stops)
        test_sents, test_spans = np.array(test_sents), np.array(test_spans)
        test_y = np.array([label2id[label] for label in test_data[args.label].values])
        test0_dataset = get_dataset(tokenizer, test_sents, test_spans, test_y)

    for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
        train, dev = next(StratifiedKFold(10, shuffle=True).split(
            np.zeros(len(train)), y[train]))

        train_dataset = get_dataset(tokenizer, sents[train], spans[train], y[train])
        dev_dataset = get_dataset(tokenizer, sents[dev], spans[dev], y[dev])
        test_dataset = get_dataset(tokenizer, sents[test], spans[test], y[test])

        model = AutoModelForSequenceClassification.from_pretrained(
                args.modelname, num_labels=len(set(y))
            ).to(device)
        # this is needed, since we have expanded the tokenizer to incorporate
        # the target special token [TGT]
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=4.5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=args.epochs,
            weight_decay=0.1,
            do_eval=True,
            save_strategy='epoch',
            evaluation_strategy="epoch",
            load_best_model_at_end=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            # early stopping
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

        trainer.train()
        # these are actually the logits
        preds, _, _ = trainer.predict(test_dataset)
        scores = np.max(softmax(preds, axis=1), axis=1)
        preds = np.argmax(preds, axis=1)
        preds = [id2label[i] for i in preds]

        folds.append(pd.DataFrame({
            'fold': fold, 
            'test': test,
            'trues': [id2label[y[i]] for i in test],
            'scores': scores, 
            'preds': preds}))

        if test_data is not None:
            preds, _, _ = trainer.predict(test0_dataset)
            scores = np.max(softmax(preds, axis=1), axis=1)
            preds = np.argmax(preds, axis=1)
            preds = [id2label[i] for i in preds]
            test_folds.append(pd.DataFrame({
                'fold': fold, 
                'preds': preds, 
                'scores': scores,
                'trues': [id2label[l] for l in test_y]
            }))

    prefix, suffix = os.path.splitext(args.input_file)
    if args.infix:
        prefix = ''.join([prefix, args.infix])
    pd.concat(folds).to_parquet(''.join([prefix, '.finetune.results.parquet']))

    if test_data is not None:
        pd.concat(test_folds).to_parquet(''.join([prefix, '.finetune.test-results.parquet']))
