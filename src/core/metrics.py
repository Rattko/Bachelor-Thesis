from autosklearn.metrics import precision, recall, f1, make_scorer, Scorer
from sklearn.metrics import average_precision_score, roc_auc_score

pr_auc = make_scorer(
    name='pr_auc',
    score_func=average_precision_score,
    optimum=1.0,
    worst_possible_result=0.0,
    greater_is_better=True,
    needs_threshold=True
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
