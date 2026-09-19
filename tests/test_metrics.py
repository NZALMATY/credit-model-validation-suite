import numpy as np
from src.metrics import calculate_gini, calculate_ks, calculate_psi


def test_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    assert calculate_gini(y_true, y_prob) == 1.0
    assert calculate_ks(y_true, y_prob) == 1.0
