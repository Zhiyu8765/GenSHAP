import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def PerformancePlot_Removal(vals_dict, X_train, y_train, X_test, y_test, a, b, order='descending', random_iters=10,
                            seed=None, save_path=None):
    '''
    Plots the effect of cumulatively removing data points based on TMC Shapley, LOO, and a Random baseline.

    Parameters:
        vals_dict: Dictionary of valuation vectors, e.g., {'TMC Shapley': vals1, 'LOO': vals2}
        X_train, y_train: Training data
        X_test, y_test: Test data
        a: Fraction of data removed at each step
        b: Total fraction of data to remove
        order: 'ascending' or 'descending'
        random_iters: Number of iterations to average for random baseline
        seed: Optional seed for reproducibility
        save_path: Optional path to save the plot

    Returns:
        removal_percentages: List of percentages of training data removed
        results: Dictionary of accuracy curves per method
    '''
    if seed is not None:
        np.random.seed(seed)

    n = len(X_train)
    total_remove_count = int(np.floor(b * n))
    step_remove_count = max(1, int(np.floor(a * n)))
    removal_counts = list(range(0, total_remove_count + 1, step_remove_count))
    removal_percentages = [(count / n) * 100 for count in removal_counts]
    results = {}

    def compute_curve(ordering):
        curve = []
        for r_count in removal_counts:
            keep_indices = ordering[r_count:]
            if len(keep_indices) == 0:
                curve.append(0.0)
                continue
            X_sub = X_train[keep_indices]
            y_sub = y_train[keep_indices]
            unique_classes = np.unique(y_sub)
            if len(unique_classes) < 2:
                dummy_pred = np.full_like(y_test, fill_value=unique_classes[0])
                acc = accuracy_score(y_test, dummy_pred)
            else:
                model = LogisticRegression(max_iter=5000, random_state=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_sub, y_sub)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            curve.append(acc)
        return np.array(curve)

    # Compute curves for TMC and LOO
    for name, val in vals_dict.items():
        sorted_indices = np.argsort(val) if order == 'ascending' else np.argsort(val)[::-1]
        results[name] = compute_curve(sorted_indices)

    # Random baseline
    random_curves = []
    for _ in range(random_iters):
        random_order = np.random.permutation(n)
        random_curves.append(compute_curve(random_order))
    results['Random'] = np.mean(random_curves, axis=0)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.xlabel('Percentage of training data removed (%)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    for name, curve in results.items():
        plt.plot(removal_percentages, curve * 100, lw=3, marker='o', label=name)
    plt.legend(fontsize=14)
    plt.title('Performance vs. Data Removal', fontsize=18)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    return removal_percentages, results