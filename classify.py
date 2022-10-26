

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='CSV file with embeddings in it')
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

    prefix, suffix = os.path.splitext(args.input_file)
    pd.concat(folds).to_parquet(''.join([prefix, '.{}.results'.format(args.model), suffix]))
