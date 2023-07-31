#!/usr/bin/env python3

import lzma
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from core.dataset import Dataset
from core.utils import calculate_imbalance


def main(directory: str, file_path: str) -> None:
    with open(file_path, 'r') as data_file:
        data = pd.read_csv(data_file)
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    target = data[' Label']
    data = data.drop(' Label', axis=1)

    # Convert multi-class dataset to binary
    target[target != 'BENIGN'] = 'MALICIOUS'

    pipeline = Pipeline([
        ('transformer', ColumnTransformer(
            [
                (
                    'ordinal_encoder',
                    OrdinalEncoder(encoded_missing_value=np.nan),
                    data.select_dtypes(include='int64').columns
                ),
                (
                    'standard_scaler',
                    StandardScaler(),
                    data.select_dtypes(include='float64').columns
                )
            ],
            remainder='passthrough', n_jobs=-1
        )),
        ('imputer', SimpleImputer())
    ])

    data = pipeline.fit_transform(data)
    target = LabelEncoder().fit_transform(target)

    # Select majority samples to keep
    majority_to_keep = np.random.choice(
        np.where(target == 0)[0], size=int(sum(target == 0) / 10), replace=False
    )

    # Select minority samples to keep
    minority_to_keep = np.random.choice(
        np.where(target == 1)[0], size=int(sum(target == 1) / 100), replace=False
    )

    data = np.concatenate((data[majority_to_keep], data[minority_to_keep]), axis=0)
    target = np.concatenate((target[majority_to_keep], target[minority_to_keep]), axis=0)

    dataset = Dataset(
        dataset_id='ids',
        name='CIC-IDS-2017',
        size=len(data),
        majority_size=sum(target == 0),
        minority_size=sum(target == 1),
        imbalance=calculate_imbalance(target),
        url='https://www.unb.ca/cic/datasets/ids-2017.html',
        data=data,
        target=target
    )

    with lzma.open(f'{directory}/{dataset.dataset_id}.pickle', mode='wb') as dataset_file:
        pickle.dump(dataset, dataset_file)


if __name__ == '__main__':
    main('datasets', 'datasets/ids/data.csv')
