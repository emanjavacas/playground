

import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    # data = pd.read_parquet('../geur/reuk_300.embeddings.parquet')

    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label]])
    X = np.stack(data['embedding'].values)

    cv = StratifiedKFold(10, shuffle=True, random_state=135)
    folds = []

    for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):

        if args.model == 'SVC':
            clf = make_pipeline(StandardScaler(), LinearSVC(C=1e-5))
        elif args.model == 'KNN':
            clf = KNeighborsClassifier()
        else:
            raise ValueError

        clf.fit(X[train], y[train])
        preds = [id2label[id] for id in clf.predict(X[test])]

        folds.append(pd.DataFrame({'fold': fold, 'test': test, 'preds': preds}))

    prefix, suffix = os.path.splitext(args.input_file)
    pd.concat(folds).to_parquet(''.join([prefix, '.{}.results'.format(args.model), suffix]))

    # # Hyperparameter Search
    # if args.model == 'SVC':
    #     opt = RandomizedSearchCV(
    #         make_pipeline(StandardScaler(), LinearSVC(random_state=153)),
    #         {
    #             'linearsvc__C': loguniform(1e-6, 1e+6),
    #             'linearsvc__class_weight': [None, 'balanced']
    #         },
    #         scoring='f1_macro',
    #         cv=cv,
    #         n_iter=10,
    #         n_jobs=-1,
    #         verbose=1000,
    #         random_state=153,
    #         refit=False,
    #         return_train_score=True)

    # elif args.model == 'KNN':
    #     opt = GridSearchCV(
    #         KNeighborsClassifier(),
    #         {
    #             'n_neighbors': [3,5,11,19],
    #             'weights': ['uniform', 'distance'],
    #             'metric': ['manhattan', 'euclidean']
    #         },
    #         scoring='rand_score',
    #         cv=cv,
    #         n_jobs=-1,
    #         refit=False,
    #         verbose=1000)
    # else:
    #     raise ValueError

    # opt.fit(X, y)