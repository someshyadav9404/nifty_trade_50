from numpy.random import default_rng
import numpy as np

def bootstrap_sharpe_test(ret_a, ret_b, n_boot=1000, seed=42):

    rng = default_rng(seed)

    n = min(len(ret_a), len(ret_b))

    ret_a = ret_a[:n]
    ret_b = ret_b[:n]

    sharpe_diffs = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        sample_a = ret_a[idx]
        sample_b = ret_b[idx]

        sharpe_a = np.mean(sample_a) / (np.std(sample_a)+1e-9) * np.sqrt(252)
        sharpe_b = np.mean(sample_b) / (np.std(sample_b)+1e-9) * np.sqrt(252)

        sharpe_diffs.append(sharpe_a - sharpe_b)

    sharpe_diffs = np.array(sharpe_diffs)

    p_value = np.mean(sharpe_diffs <= 0)

    return np.mean(sharpe_diffs), p_value