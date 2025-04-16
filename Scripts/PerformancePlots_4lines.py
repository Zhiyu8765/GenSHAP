import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def PerformancePlot_Removal(vals, X_train, y_train, X_test, y_test, a, b, order='descending', random_iters=10,
                            save_path=None):
    '''
    Plots the effect of cumulatively removing data points based on one or more valuation methods.

    Parameters:
        vals: Data valuation arrays (a single array or list/tuple of arrays)
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features
        y_test: Test data labels
        a: Fraction of data points removed in each step (e.g., 0.05 for 5%)
        b: Total fraction of data to remove (e.g., 0.4 for 40%)
        order: Removal order ('descending' removes highest-valued first; 'ascending' removes lowest-valued first)
        random_iters: Number of iterations to average over for the random baseline
        save_path: (Optional) File path to save the generated plot.

    Returns:
        removal_percentages: List of percentages of training data removed at each step.
        results: Dictionary mapping method names to the corresponding test accuracy curve.
    '''
    if not isinstance(vals, (list, tuple)):
        vals = [vals]

    n = len(X_train)
    total_remove_count = int(np.floor(b * n))
    step_remove_count = max(1, int(np.floor(a * n)))
    removal_counts = list(range(0, total_remove_count + 1, step_remove_count))
    removal_percentages = [(count / n) * 100 for count in removal_counts]
    results = {}

    def compute_curve(ordering):
        curve = []
        for r_count in removal_counts:
            # Cumulatively remove the first r_count indices
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
                curve.append(acc)
                continue
            model = LogisticRegression(max_iter=5000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            # Predict on X_test so that y_pred and y_test have consistent lengths.
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            curve.append(acc)
        return np.array(curve)

    # Compute performance curve for each valuation method.
    for i, val in enumerate(vals):
        if order == 'ascending':
            sorted_indices = np.argsort(val)
        else:
            sorted_indices = np.argsort(val)[::-1]
        results[f'valuation {i + 1}'] = compute_curve(sorted_indices)

    # Compute the random baseline curve by averaging over random orders.
    random_curves = []
    for _ in range(random_iters):
        random_order = np.random.permutation(n)
        random_curves.append(compute_curve(random_order))
    random_curve = np.mean(random_curves, axis=0)
    results['Random Baseline'] = random_curve

    # Rename the curves if exactly three valuation arrays are provided.
    if len(vals) == 3:
        results_renamed = {}
        results_renamed['TMC Shapley'] = results.pop('valuation 1')
        results_renamed['G Shapley'] = results.pop('valuation 2')
        results_renamed['LOO'] = results.pop('valuation 3')
        results_renamed['Random'] = results.pop('Random Baseline')
        results = results_renamed

    # Plot the results.
    plt.figure(figsize=(8, 8))
    plt.xlabel('Percentage of training data removed (%)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    for key, curve in results.items():
        plt.plot(removal_percentages, curve * 100, lw=3, marker='o', label=key)
    plt.legend(fontsize=14)
    plt.title('Performance vs. Data Removal', fontsize=18)
    plt.grid(True)

    # Save the figure if a save_path is provided.
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    return removal_percentages, results