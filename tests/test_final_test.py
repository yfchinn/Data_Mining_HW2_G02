import numpy as np
from sklearn.metrics import average_precision_score
from evaluate_final_test import ap_cache,weighted_ap


def test_bootstrap_ap_matches_sklearn_with_ties_and_zero_weights():
    y=np.array([0,1,0,1,0,1])
    scores=np.array([.8,.8,.2,.4,.4,.1])
    cache=ap_cache(y,scores)
    for weights in (np.ones(6),np.array([0,2,1,0,0,3]),np.array([3,0,0,1,1,1])):
        expected=average_precision_score(y,scores,sample_weight=weights)
        assert np.isclose(weighted_ap(cache,weights),expected)
