import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from collections.abc import Mapping

def PerformancePlot_Removal(vals, X_train, y_train, X_test, y_test,
                            a, b, *, order='descending', random_iters=10,
                            ylim = None, Names = None, save_path=None):
    '''
    Plots the effect of cumulatively removing data points based various data valuation methods.

    Parameters:
        vals: Data values
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features
        y_test: Test data labels
        a: Fraction of data points removed in each step (e.g. 0.05)
        b: Total fraction of data to remove (e.g. 0.4 )
        ylim: Range on y-axis.
        Names: Legends of plot.
        order: Removal order ('descending' removes highest-valued first; 'ascending' removes lowest-valued first)
        random_iters: Number of iterations to average over for the random baseline
        save_path: (Optional) File path to save the generated plot.

    Returns:
        removal_percentages: List of percentages of training data removed at each step.
        results: Dictionary mapping method names to the corresponding test accuracy curve.
    '''
    # Read in arrays of data values
    if isinstance(vals, Mapping):
        labels, val_arrays = list(vals.keys()), list(vals.values())
    else:
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        val_arrays = list(vals)

        # Write down legend names
        labels = (list(names) if names is not None else
                  [getattr(v, 'name', None) for v in val_arrays])
        labels = [lbl if lbl else f'valuation {i + 1}'
                  for i, lbl in enumerate(labels)]

    # Define range and steps of removal
    n = len(X_train)
    total_remove = int(np.floor(b * n))
    step_remove  = max(1, int(np.floor(a * n)))
    removal_counts      = list(range(0, total_remove + 1, step_remove))
    removal_percentages = [100 * r / n for r in removal_counts]

    def _curve(ordering):
        curve = []
        for r in removal_counts:
            keep = ordering[r:]
            if len(keep) == 0:
                curve.append(0.0)
                continue
            X_sub, y_sub = X_train[keep], y_train[keep]
            if len(np.unique(y_sub)) < 2:
                dummy = np.full_like(y_test, y_sub[0])
                curve.append(accuracy_score(y_test, dummy))
                continue
            model = LogisticRegression(max_iter=5000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            curve.append(accuracy_score(y_test, model.predict(X_test)))
        return np.asarray(curve)

    # Compute performance curve for each valuation method.
    results = {}
    for lbl, val in zip(labels, val_arrays):
        order_idx = np.argsort(val)
        if order != 'ascending':
            order_idx = order_idx[::-1]
        results[lbl] = _curve(order_idx)

    # Random Baseline
    random_curves = [_curve(np.random.permutation(n)) for _ in range(random_iters)]
    results['Random'] = np.mean(random_curves, axis=0)

    # Rename the curves if exactly three valuation arrays are provided.
    plt.figure(figsize=(8, 8))
    plt.xlabel('Percentage of training data removed (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)

    for key, curve in results.items():
        plt.plot(removal_percentages, curve * 100,
                 lw=3, marker='o', label=key)

    plt.title('Performance vs. Data Removal', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    if ylim is not None:
        plt.ylim(*ylim)  # ylim is already in %

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    return removal_percentages, results