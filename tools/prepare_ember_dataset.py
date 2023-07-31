#!/usr/bin/env python3

import lzma
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from core.dataset import Dataset
from core.utils import calculate_imbalance


def read_vectorized_features(directory: str) -> tuple[np.ndarray, np.ndarray]:
    data_train_path = os.path.join(directory, 'ember/data_train.dat')
    target_train_path = os.path.join(directory, 'ember/target_train.dat')

    target_train = np.memmap(target_train_path, dtype=np.float32, mode='r')

    ncols, nrows = 2381, target_train.shape[0]

    data_train = np.memmap(data_train_path, dtype=np.float32, mode='r', shape=(nrows, ncols))

    data_test_path = os.path.join(directory, 'ember/data_test.dat')
    target_test_path = os.path.join(directory, 'ember/target_test.dat')

    target_test = np.memmap(target_test_path, dtype=np.float32, mode='r')

    ncols, nrows = 2381, target_test.shape[0]

    data_test = np.memmap(data_test_path, dtype=np.float32, mode='r', shape=(nrows, ncols))

    return (
        np.concatenate((data_train, data_test), axis=0),
        np.concatenate((target_train, target_test), axis=0)
    )


def main(directory: str) -> None:
    data, target = read_vectorized_features(directory)

    # Get rid of unlabeled data
    data = data[target != -1]
    target = target[target != -1]

    # Select minority samples to keep
    minority_to_keep = np.random.choice(
        np.where(target == 1)[0], size=int(sum(target == 1) / 15), replace=False
    )

    # Select majority samples to keep
    majority_to_keep = np.random.choice(
        np.where(target == 0)[0], size=int(sum(target == 0) / 2), replace=False
    )

    data = np.concatenate((data[majority_to_keep], data[minority_to_keep]), axis=0)
    target = np.concatenate((target[majority_to_keep], target[minority_to_keep]), axis=0)

    # Reduce the number of features
    clf = RandomForestClassifier()
    clf.fit(data, target)

    feature_selector = SelectFromModel(clf, prefit=True, max_features=300)
    data = feature_selector.transform(data)

    dataset = Dataset(
        dataset_id='ember',
        name='Ember',
        size=len(data),
        majority_size=sum(target == 0),
        minority_size=sum(target == 1),
        imbalance=calculate_imbalance(target),
        url='https://github.com/elastic/ember',
        data=data,
        target=target
    )

    with lzma.open(f'{directory}/{dataset.dataset_id}.pickle', mode='wb') as dataset_file:
        pickle.dump(dataset, dataset_file)


if __name__ == '__main__':
    main('datasets')
