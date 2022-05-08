#!/usr/bin/env python3

import argparse
import lzma
import os
import pickle
import sys

import numpy as np
import openml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from tqdm import tqdm

from core.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--classes', type=int, default=2, help='Number of classes in a dataset'
)
parser.add_argument(
    '--directory', type=str, default='./datasets', help='Path to the destination directory'
)
parser.add_argument(
    '--max_missing_values', type=float, default=0.2,
    help='Maximum percentage of samples with missing values'
)
parser.add_argument(
    '--min_imbalance_ratio', type=int, default=10,
    help='Minimum ratio between a majority and minority class'
)
parser.add_argument(
    '--min_instances', type=int, default=5000, help='Minimum size of a dataset'
)
parser.add_argument(
    '--wipe_datasets', action='store_true', default=False, help='Wipe already downloaded datasets'
)

openml.config.apikey = '47e99eef7c8b1b776c8bbadcd7763361'
openml.config.cache_directory = os.path.expanduser('./.openml_cache')

def list_datasets(
    min_instances: int, max_missing_values: float, classes: int, min_imbalance_ratio: int
) -> list[Dataset]:
    datasets_info = openml.datasets.list_datasets(output_format='dataframe')

    datasets_info['ImbalanceRatio'] = (
        datasets_info['MajorityClassSize'] / datasets_info['MinorityClassSize']
    )
    datasets_info['PercentageOfInstancesWithMissingValues'] = (
        datasets_info['NumberOfInstancesWithMissingValues'] / datasets_info['NumberOfInstances']
    )

    # Filter datasets based on the given conditions and sort the results by 'ImbalanceRatio'
    datasets_info = datasets_info[
        (datasets_info['NumberOfInstances'] > min_instances) &
        (datasets_info['PercentageOfInstancesWithMissingValues'] <= max_missing_values) &
        (datasets_info['NumberOfClasses'] == classes) &
        (datasets_info['ImbalanceRatio'] >= min_imbalance_ratio)
    ].sort_values(by='ImbalanceRatio')

    # Drop entries which have the same number of total, majority and minority class instances
    datasets_info = datasets_info.drop_duplicates(
        ['NumberOfInstances', 'MajorityClassSize', 'MinorityClassSize']
    )

    columns = [
        'did', 'name', 'NumberOfInstances', 'MajorityClassSize',
        'MinorityClassSize', 'ImbalanceRatio'
    ]

    return [Dataset(*data) for _, data in datasets_info[columns].iterrows()]

def get_citations(datasets: list[Dataset]) -> list[tuple[str, str, str]]:
    datasets_info = openml.datasets.get_datasets([dataset.dataset_id for dataset in datasets])

    return [
        (dataset.name, dataset.citation, dataset.openml_url)
        for dataset in datasets_info if dataset.citation
    ]

def get_licences(datasets: list[Dataset]) -> list[tuple[str, str, str]]:
    datasets_info = openml.datasets.get_datasets([dataset.dataset_id for dataset in datasets])

    return [(dataset.name, dataset.licence, dataset.openml_url) for dataset in datasets_info]

def get_dataset(dataset: Dataset, directory: str) -> None:
    dataset_info = openml.datasets.get_dataset(dataset.dataset_id)

    dataset.url = dataset_info.openml_url

    data, target, categorical_indicator, _ = dataset_info.get_data(
        dataset_format='dataframe', target=dataset_info.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)

    # OpenML returns an incorrect `categorical_indicator` for the dataset with ID 42252
    # and this is the most straighforward way to avoid an exception down the road.
    # https://new.openml.org/d/42252
    if dataset.dataset_id == 42252:
        categorical_indicator[0] = True

    continuous_predictors = ~categorical_indicator
    ordinal_predictors = (data.nunique() > 64) & categorical_indicator
    categorical_predictors = categorical_indicator ^ ordinal_predictors

    assert all(continuous_predictors | ordinal_predictors | categorical_predictors), 'Oi!'

    pipeline = Pipeline([
        ('transformer', ColumnTransformer(
            [
                ('one_hot_encoder', OneHotEncoder(sparse=False), categorical_predictors),
                ('ordinal_encoder', OrdinalEncoder(), ordinal_predictors),
                ('standard_scaler', StandardScaler(), continuous_predictors)
            ],
            remainder='passthrough', n_jobs=-1
        )),
        ('imputer', SimpleImputer())
    ])

    dataset.data = pipeline.fit_transform(data)
    dataset.target = LabelEncoder().fit_transform(target)

    # Labels are swapped, i.e. the positive class has label 0 and the negative class has label 1
    if len(dataset.target[dataset.target == 1]) > len(dataset.target[dataset.target == 0]):
        dataset.target = np.where(
            (dataset.target == 0) | (dataset.target == 1),
            dataset.target ^ 1, dataset.target
        )

    with lzma.open(f'{directory}/{dataset.dataset_id}.pickle', mode='wb') as dataset_file:
        pickle.dump(dataset, dataset_file)

def main(args: argparse.Namespace, blacklist: list[str]) -> None:
    os.makedirs(args.directory, exist_ok=True)

    datasets = list_datasets(
        args.min_instances, args.max_missing_values, args.classes, args.min_imbalance_ratio
    )
    datasets = [dataset for dataset in datasets if dataset.name not in blacklist]
    downloaded_datasets = [
        x.removesuffix('.pickle') for x in os.listdir(args.directory)
        if x.endswith('.pickle') and x != 'unsw.pickle'
    ]

    if args.wipe_datasets:
        for file in downloaded_datasets:
            os.remove(f'{args.directory}/{file}.pickle')

        downloaded_datasets = []

    for dataset in tqdm(datasets, desc='Downloading datasets', leave=False):
        if str(dataset.dataset_id) in downloaded_datasets:
            continue

        try:
            get_dataset(dataset, args.directory)
        except Exception as exc:
            print(dataset, file=sys.stderr)
            print(exc, file=sys.stderr)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args, blacklist=['20_newsgroups.drift'])
