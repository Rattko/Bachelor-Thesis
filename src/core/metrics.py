from autosklearn.metrics import accuracy, balanced_accuracy, f1, log_loss
from autosklearn.metrics import make_scorer, precision, recall, Scorer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

pr_auc = make_scorer(
    name='pr_auc',
    score_func=average_precision_score,
    optimum=1.0,
    worst_possible_result=0.0,
    greater_is_better=True,
    needs_threshold=True
)

matthews_corr_coef = make_scorer(
    name='matthews_corr_coef',
    score_func=matthews_corrcoef,
    optimum=1.0,
    worst_possible_result=-1.0,
    greater_is_better=True,
    needs_threshold=False
)

roc_auc = make_scorer(
    name='roc_auc',
    score_func=roc_auc_score,
    optimum=1.0,
    worst_possible_result=0.0,
    greater_is_better=True,
    needs_threshold=True
)

def partial_roc_auc(max_fpr: float) -> Scorer:
    return make_scorer(
        name='partial_roc_auc',
        score_func=roc_auc_score,
        optimum=1.0,
        worst_possible_result=0.0,
        greater_is_better=True,
        needs_threshold=True,
        **{'max_fpr': max_fpr}
    )
