
"""
classify.py: This script performs classification on the provided train file, and
inference on the provided test file. For more information on how to run this 
script, run the command ‘python classify.py --help’ in the command line.
"""

__author__    = "Enrique Manjavacas"
__copyright__ = "Copyright 2022, Enrique Manjavacas"
__license__   = "GPL"
__version__   = "1.0.1"


import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.fixes import loguniform
import numpy as np
import pandas as pd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
    "This file performs classification on top of frozen embeddings. The embeddings can be "
    "first extracted with the preprocess.py file."
    "Evaluation is done with Cross-Validation. After training the cross-validation "
    "results on each fold are stored to a {model}.results.parquet file."
    "If a test-file was provided, then the model trained in each iteration is used "
    "for prediction on the test file and the results are output to a file with the "
    "suffix: {model}.test-results.parquet")
    parser.add_argument('--input-file', required=True, help='CSV file with embeddings in it')
    parser.add_argument('--test-file')
    parser.add_argument('--label', required=True)
    parser.add_argument('--model', default='SVC')
    args = parser.parse_args()

    data = pd.read_parquet(args.input_file) 

    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label].values])
    X = np.stack(data['embedding'].values)
    # print(np.where(np.isnan(X).sum(1)))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=135)
    folds = []

    test_data = None
    test_folds = []
    if args.test_file:
        test_data = pd.read_parquet(args.test_file)
        test_y = np.array([label2id[label] for label in test_data[args.label].values])
        test_X = np.stack(test_data['embedding'].values)

    for fold, (train, test) in enumerate(cv.split(X, y)):

        if args.model == 'SVC':
            opt = RandomizedSearchCV(
                make_pipeline(StandardScaler(), LinearSVC()),
                {
                    'linearsvc__C': loguniform(1e-6, 1e+6),
                    'linearsvc__class_weight': [None, 'balanced']
                },
                scoring='f1_macro',
                cv=StratifiedKFold(5, shuffle=True),
                n_iter=10,
                n_jobs=-1,
                verbose=1000,
                random_state=153,
                refit=False,
                return_train_score=True
            ).fit(X[train], y[train])

            clf = make_pipeline(
                StandardScaler(), 
                LinearSVC(C=opt.best_params_['linearsvc__C'],
                          class_weight=opt.best_params_['linearsvc__class_weight']))

        elif args.model == 'KNN':
            opt = GridSearchCV(
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3,5,11,19],
                    'weights': ['uniform', 'distance'],
                    'metric': ['manhattan', 'euclidean']
                },
                scoring='rand_score',
                cv=StratifiedKFold(5, shuffle=True),
                n_jobs=-1,
                refit=False,
                verbose=1000
            ).fit(X[train], y[train])
            
            clf = KNeighborsClassifier(
                metric=opt.best_params_['metric'],
                n_neighbors=opt.best_params_['n_neighbors'],
                weights=opt.best_params_['weights'])

        else:
            raise ValueError

        clf.fit(X[train], y[train])

        folds.append(pd.DataFrame({
            'fold': fold, 
            'test': test, 
            'preds': [id2label[id] for id in clf.predict(X[test])], 
            'trues': [id2label[y[i]] for i in test]
        }))

        if test_data is not None:
            test_folds.append(pd.DataFrame({
                'fold': fold, 
                'preds': [id2label[id] for id in clf.predict(test_X)], 
                'trues': [id2label[l] for l in test_y]
            }))

    prefix, suffix = os.path.splitext(args.input_file)
    pd.concat(folds).to_parquet(''.join([prefix, '.{}.results'.format(args.model), suffix]))

    if test_data is not None:
        pd.concat(test_folds).to_parquet(''.join([prefix, '.{}.test-results'.format(args.model), suffix]))
