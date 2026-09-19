import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    df = pd.DataFrame({"target": y_true, "prob": y_prob})
    df = df.sort_values("prob", ascending=False)
    df["cum_event"] = (df["target"] == 1).cumsum() / (
        df["target"] == 1
    ).sum()
    df["cum_non_event"] = (df["target"] == 0).cumsum() / (
        df["target"] == 0
    ).sum()
    return float(np.max(np.abs(df["cum_event"] - df["cum_non_event"])))


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10
) -> float:
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.percentile(expected, quantiles * 100)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )
