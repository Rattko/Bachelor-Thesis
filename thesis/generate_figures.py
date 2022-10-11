#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.under_sampling import AllKNN, ClusterCentroids, CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection, TomekLinks
from matplotlib.pyplot import Figure
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


BLUE = '#00BEFF'
YELLOW = '#ECB53A'
BROWN = '#5E4947'
RED = '#B52440'


def generate_data(
    majority_size: int, minority_size: int, random_state: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    majority_distribution = multivariate_normal([-0.8, 0], [[1, 0.1], [0.3, 1]])
    minority_distribution = multivariate_normal([1.3, -0.1], [[1, 0.6], [-0.3, 1.2]])

    data_majority = majority_distribution.rvs(size=majority_size, random_state=random_state)
    data_minority = minority_distribution.rvs(size=minority_size, random_state=random_state)

    return (
        np.vstack((data_majority, data_minority)),
        np.hstack((np.zeros(majority_size), np.ones(minority_size)))
    )


def generate_oversampling_scatter_plots(
    plot_info: list[tuple[BaseSampler, str]], data: np.ndarray, target: np.ndarray
) -> Figure:
    fig, axes = plt.subplots(nrows=len(plot_info) // 2, ncols=2, figsize=(16, len(plot_info) * 3))

    for (resampler, title), axis in zip(plot_info, axes):
        data_gen, _ = resampler.fit_resample(data, target)

        data_new = data_gen[[ind for ind, x in enumerate(data_gen) if x not in data]]

        axis.scatter(data_new[:, 0], data_new[:, 1], c=BROWN, label='Generated')
        axis.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
        axis.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
        axis.set_title(title)
        axis.legend(loc='upper left')

    return fig


def generate_undersampling_scatter_plots(
    resampler: BaseSampler, data: np.ndarray, target: np.ndarray
) -> Figure:
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(24, 6)
    )

    data_keep, target_keep = resampler.fit_resample(data, target)

    # Original dataset
    ax1.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax1.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax1.set_title('Original Dataset')
    ax1.legend(loc='upper left')

    # Samples to delete are highlighted
    ax2.scatter(data[target == 0, 0], data[target == 0, 1], c=BROWN, label='Marked')
    ax2.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax2.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax2.tick_params(axis='both', reset=True, top=False, right=False)
    ax2.set_title('Highlighted Original Dataset')
    ax2.legend(loc='upper left')

    # Resampled dataset
    ax3.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax3.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax3.tick_params(axis='both', reset=True, top=False, right=False)
    ax3.set_title('Resampled Dataset')
    ax3.legend(loc='upper left')

    return fig


def resampling_using_smote(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    resampler = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)

    data_gen, _ = resampler.fit_resample(data, target)

    data_new = data_gen[[ind for ind, x in enumerate(data_gen) if x not in data]]

    axis.scatter(data_new[:, 0], data_new[:, 1], c=BROWN, label='Generated')
    axis.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    axis.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    axis.set_title('SMOTE')
    axis.legend(loc='upper left')

    if save:
        plt.savefig('./figures/smote.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def random_oversampling_vs_rose(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    resampler = RandomOverSampler(sampling_strategy=1.0, shrinkage=0.0, random_state=1)
    data_resampled, _ = resampler.fit_resample(data, target)
    data_new = data_resampled[len(data):]

    ax1.scatter(data_new[:, 0], data_new[:, 1], s=90, c=BROWN, label='Generated')
    ax1.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax1.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax1.set_title('Random Oversampling without Smoothing')
    ax1.legend(loc='upper left')

    resampler = RandomOverSampler(sampling_strategy=1.0, shrinkage=0.2, random_state=1)
    data_resampled, _ = resampler.fit_resample(data, target)
    data_new = data_resampled[len(data):]

    ax2.scatter(data_new[:, 0], data_new[:, 1], c=BROWN, label='Generated')
    ax2.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax2.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax2.set_title('Random Oversampling with Smoothing')
    ax2.legend(loc='upper left')

    if save:
        plt.savefig('./figures/random_oversampling_vs_rose.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def smote_vs_borderline_smote(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)
    borderline_smote = BorderlineSMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)

    fig = generate_oversampling_scatter_plots(
        [(smote, 'SMOTE'), (borderline_smote, 'BorderlineSMOTE')], data, target
    )

    if save:
        plt.savefig('./figures/smote_vs_borderlinesmote.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def smote_vs_svm_smote(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)
    svm_smote = SVMSMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)

    fig = generate_oversampling_scatter_plots(
        [(smote, 'SMOTE'), (svm_smote, 'SVM SMOTE')], data, target
    )

    if save:
        plt.savefig('./figures/smote_vs_svmsmote.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def smote_vs_kmeans_smote(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)
    kmeans_smote = KMeansSMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)

    fig = generate_oversampling_scatter_plots(
        [(smote, 'SMOTE'), (kmeans_smote, 'KMeans SMOTE')], data, target
    )

    if save:
        plt.savefig('./figures/smote_vs_kmeanssmote.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def smote_vs_adasyn(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=1)
    adasyn = ADASYN(sampling_strategy='minority', n_jobs=-1, random_state=1)

    fig = generate_oversampling_scatter_plots(
        [(smote, 'SMOTE'), (adasyn, 'ADASYN')], data, target
    )

    if save:
        plt.savefig('./figures/smote_vs_adasyn.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def cnn(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    resampler = CondensedNearestNeighbour(n_jobs=-1, random_state=1)

    fig = generate_undersampling_scatter_plots(resampler, data, target)

    if save:
        plt.savefig('./figures/cnn.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def enn(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    resampler = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode', n_jobs=-1)

    fig = generate_undersampling_scatter_plots(resampler, data, target)

    if save:
        plt.savefig('./figures/enn.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def all_knn(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    resampler = AllKNN(n_neighbors=7, kind_sel='all', n_jobs=-1)

    fig = generate_undersampling_scatter_plots(resampler, data, target)

    if save:
        plt.savefig('./figures/all_knn.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def near_miss(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(16, 12)
    )

    # Original dataset
    ax1.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax1.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax1.tick_params(axis='both', reset=True, top=False, right=False)
    ax1.set_title('Original Dataset')
    ax1.legend(loc='upper left')

    for version, axis in zip([1, 2, 3], [ax2, ax3, ax4]):
        resampler = NearMiss(version=version, n_jobs=-1)
        data_keep, target_keep = resampler.fit_resample(data, target)

        # Resampled dataset with NearMiss
        axis.scatter(
            data_keep[target_keep == 0, 0],
            data_keep[target_keep == 0, 1],
            c=BLUE, label='Majority'
        )
        axis.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
        axis.tick_params(axis='both', reset=True, top=False, right=False)
        axis.set_title(f'Resampled Dataset using Near Miss {version}')
        axis.legend(loc='upper left')

    if save:
        plt.savefig('./figures/near_miss.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def tomek_links(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    resampler = TomekLinks(n_jobs=-1)

    fig = generate_undersampling_scatter_plots(resampler, data, target)

    if save:
        plt.savefig('./figures/tomek_links.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def oss(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    # Add a noisy sample point
    data = np.append(data, [[2.2, 1.5]], axis=0)
    target = np.append(target, [[0]])

    oss_resampler = OneSidedSelection(n_jobs=-1, random_state=1)
    tomek_links_resampler = TomekLinks()

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(24, 6)
    )

    data_keep, target_keep = oss_resampler.fit_resample(data, target)
    data_keep_tomek, target_keep_tomek = tomek_links_resampler.fit_resample(data, target)

    # Original dataset
    ax1.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax1.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax1.set_title('Original Dataset')
    ax1.legend(loc='upper left')

    # Samples to delete are highlighted
    ax2.scatter(data[target == 0, 0], data[target == 0, 1], c=RED, label='Noisy Data')
    ax2.scatter(
        data_keep_tomek[target_keep_tomek == 0, 0],
        data_keep_tomek[target_keep_tomek == 0, 1],
        c=BROWN, label='Redundant Data'
    )
    ax2.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax2.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax2.tick_params(axis='both', reset=True, top=False, right=False)
    ax2.set_title('Highlighted Original Dataset')
    ax2.legend(loc='upper left')

    # Resampled dataset
    ax3.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax3.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax3.tick_params(axis='both', reset=True, top=False, right=False)
    ax3.set_title('Resampled Dataset')
    ax3.legend(loc='upper left')

    if save:
        plt.savefig('./figures/oss.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def ncl(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    ncl_resampler = NeighbourhoodCleaningRule(kind_sel='mode', n_jobs=-1)
    enn_resampler = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode', n_jobs=-1)

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(24, 6)
    )

    data_keep, target_keep = ncl_resampler.fit_resample(data, target)
    data_keep_enn, target_keep_enn = enn_resampler.fit_resample(data, target)

    # Original dataset
    ax1.scatter(data[target == 0, 0], data[target == 0, 1], c=BLUE, label='Majority')
    ax1.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax1.set_title('Original Dataset')
    ax1.legend(loc='upper left')

    # Samples to delete are highlighted
    ax2.scatter(data[target == 0, 0], data[target == 0, 1], c=RED, label='Marked by ENN')
    ax2.scatter(
        data_keep_enn[target_keep_enn == 0, 0],
        data_keep_enn[target_keep_enn == 0, 1],
        c=BROWN, label='Marked by Cleaning'
    )
    ax2.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax2.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax2.tick_params(axis='both', reset=True, top=False, right=False)
    ax2.set_title('Highlighted Original Dataset')
    ax2.legend(loc='lower left')

    # Resampled dataset
    ax3.scatter(
        data_keep[target_keep == 0, 0], data_keep[target_keep == 0, 1], c=BLUE, label='Majority'
    )
    ax3.scatter(data[target == 1, 0], data[target == 1, 1], c=YELLOW, label='Minority')
    ax3.tick_params(axis='both', reset=True, top=False, right=False)
    ax3.set_title('Resampled Dataset')
    ax3.legend(loc='upper left')

    if save:
        plt.savefig('./figures/ncl.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def cluster_centroids(save: bool = False) -> None:
    data, target = generate_data(80, 20)

    resampler = ClusterCentroids(random_state=1)

    fig = generate_undersampling_scatter_plots(resampler, data, target)

    if save:
        plt.savefig('./figures/cluster_centroids.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def pr_curve_plot(save: bool = False) -> None:
    data, target = generate_data(800, 200)

    train_data, test_data, train_target, test_target = train_test_split(
        data, target, stratify=target, test_size=0.25, random_state=1
    )

    clf = LogisticRegression()
    clf.fit(train_data, train_target)
    preds_proba = clf.predict_proba(test_data)

    precision, recall, _ = precision_recall_curve(test_target, preds_proba[:, 1])

    fig = plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision,
        label=f'AUC = {average_precision_score(test_target, preds_proba[:, 1]) :.3f}'
    )

    plt.title('Precision Recall Curve')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')

    if save:
        plt.savefig('./figures/pr_curve.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def roc_curve_plot(save: bool = False) -> None:
    data, target = generate_data(800, 200)

    train_data, test_data, train_target, test_target = train_test_split(
        data, target, stratify=target, test_size=0.25, random_state=1
    )

    clf = LogisticRegression()
    clf.fit(train_data, train_target)
    preds_proba = clf.predict_proba(test_data)

    fpr, tpr, _ = roc_curve(test_target, preds_proba[:, 1])

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(test_target, preds_proba[:, 1]) :.3f}')
    plt.plot(
        np.linspace(0, fpr[-1], num=100),
        np.linspace(0, tpr[-1], num=100),
        label=f'AUC = {0.5:.3f}'
    )

    plt.title('Receiver Operating Characteristic Curve')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    if save:
        plt.savefig('./figures/roc_curve.eps', bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    resampling_using_smote()
    random_oversampling_vs_rose()
    smote_vs_borderline_smote()
    smote_vs_svm_smote()
    smote_vs_kmeans_smote()
    smote_vs_adasyn()
    cnn()
    enn()
    all_knn()
    near_miss()
    tomek_links()
    oss()
    ncl()
    cluster_centroids()
    pr_curve_plot()
    roc_curve_plot()
