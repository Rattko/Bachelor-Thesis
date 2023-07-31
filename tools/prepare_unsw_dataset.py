#!/usr/bin/env python3

import lzma
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.dataset import Dataset
from core.utils import calculate_imbalance


def main(directory: str, train_file_path: str, test_file_path: str) -> None:
    with open(train_file_path, 'r') as train_file, open(test_file_path, 'r') as test_file:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        data = pd.concat((train_data, test_data), ignore_index=True)
        data.drop(['attack_cat'], axis=1, inplace=True)

    pipeline = Pipeline([
        ('transformer', ColumnTransformer(
            [
                ('one_hot_encoder', OneHotEncoder(sparse=False), ['proto', 'service', 'state'])
            ],
            remainder='passthrough', n_jobs=-1
        ))
    ])

    data = pipeline.fit_transform(data)

    # Switch target labels as we expect the minority class to have label 1
    data[:, -1] = (data[:, -1] + 1) % 2

    # Select minority samples to keep
    minority_to_keep = np.random.choice(
        np.where(data[:, -1] == 1)[0], size=int(sum(data[:, -1] == 1) / 10), replace=False
    )

    data = np.concatenate((data[data[:, -1] == 0], data[minority_to_keep]), axis=0)

    target = data[:, -1]
    data = data[:, :-1]

    dataset = Dataset(
        dataset_id='unsw',
        name='UNSW-NB15',
        size=len(data),
        majority_size=sum(target == 0),
        minority_size=sum(target == 1),
        imbalance=calculate_imbalance(target),
        url='https://research.unsw.edu.au/projects/unsw-nb15-dataset',
        data=data,
        target=target
    )

    with lzma.open(f'{directory}/{dataset.dataset_id}.pickle', mode='wb') as dataset_file:
        pickle.dump(dataset, dataset_file)


if __name__ == '__main__':
    main('datasets', 'datasets/unsw/unsw-train.csv', 'datasets/unsw/unsw-test.csv')
