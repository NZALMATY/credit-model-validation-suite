import numpy as np
from src.metrics import calculate_gini, calculate_ks, calculate_psi

if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.08, 1000)
    y_prob = np.where(
        y_true == 1,
        np.random.beta(5, 2, 1000),
        np.random.beta(2, 5, 1000),
    )
    baseline_scores = np.random.normal(0.5, 0.15, 1000)
    current_scores = np.random.normal(0.52, 0.16, 1000)

    print(f"Gini Coefficient: {calculate_gini(y_true, y_prob):.4f}")
    print(f"KS Statistic:     {calculate_ks(y_true, y_prob):.4f}")
    print(
        f"PSI Stability:    {calculate_psi(baseline_scores, current_scores):.4f}"
    )
