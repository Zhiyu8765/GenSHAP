import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from collections.abc import Mapping
from matplotlib.ticker import MaxNLocator

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
    plt.rcParams.update({
        "font.size": 28,  # base font size
        "axes.titlesize": 28,  # title font
        "axes.labelsize": 28,  # axis labels
        "xtick.labelsize": 28,  # tick labels
        "ytick.labelsize": 28,
        "legend.fontsize": 23,  # legend
    })
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
            model = LogisticRegression(max_iter=200, random_state=0)
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

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, layout='constrained')
    ax.set_xlabel('Fraction of data removed (%)')
    ax.set_ylabel('Accuracy (%)')
    for key, curve in results.items():
        ax.plot(removal_percentages, curve * 100, lw=5, marker='o', label=key)
    # plt.title('Performance vs. Data Removal', fontsize=16)

    plt.grid(True)
    # plt.legend(loc='lower left')
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.98, top=0.95)
    if ylim is not None:
        plt.ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper', nbins=3))

    if save_path is not None:
        plt.savefig(save_path, format='pdf')
    # plt.subplots_adjust(left=0.0, right=0.80, bottom=0.0, top=0.80)
    plt.show()
    return removal_percentages, results


# def PerformancePlot_Addition(vals, X_train, y_train, X_test, y_test,
#                              a, b, *, order='descending', random_iters=10,
#                              ylim=None, Names=None, save_path=None, random_state=0):
#     """
#     Addition plot: cumulatively add training points according to a ranking
#     (descending = highest-valued first; ascending = lowest-valued first)
#     and track test accuracy after each addition.
#
#     Parameters
#     ----------
#     vals : Mapping[str, array] or array-like
#         One or many valuation vectors (length n). If Mapping, keys are legend names.
#     X_train, y_train, X_test, y_test : arrays
#         Train/test features and labels.
#     a : float
#         Step fraction (e.g., 0.05 adds 5% of the data each step).
#     b : float
#         Maximum fraction of data to add (e.g., 0.4 adds up to 40%).
#     order : {'descending','ascending'}, default 'descending'
#         Ranking direction for addition.
#     random_iters : int, default 10
#         Number of random baselines to average.
#     ylim : tuple or None
#         y-axis limits in percentage, e.g., (80, 105).
#     Names : list[str] or None
#         Legend labels when vals is a list/tuple instead of a Mapping.
#     save_path : str or None
#         If provided, save the figure to this path.
#     random_state : int, default 0
#         RNG seed for random baselines.
#
#     Returns
#     -------
#     add_percentages : list[float]
#         Percent of training data used at each step.
#     results : dict[str, np.ndarray]
#         Accuracy curves (in [0,1]) for each method + Random.
#     """
#     # Parse inputs
#     if isinstance(vals, Mapping):
#         labels, val_arrays = list(vals.keys()), [np.asarray(v, dtype=float) for v in vals.values()]
#     else:
#         if not isinstance(vals, (list, tuple)):
#             vals = [vals]
#         val_arrays = [np.asarray(v, dtype=float) for v in vals]
#         labels = (list(Names) if Names is not None else
#                   [getattr(v, 'name', None) for v in val_arrays])
#         labels = [lbl if lbl else f'valuation {i + 1}' for i, lbl in enumerate(labels)]
#
#     n = X_train.shape[0]
#     assert all(len(v) == n for v in val_arrays), "All valuation arrays must match |X_train|"
#
#     # Step schedule: from 0 up to floor(b*n), in steps of floor(a*n) (at least 1)
#     total_add = int(np.floor(b * n))
#     step_add  = max(1, int(np.floor(a * n)))
#     add_counts       = list(range(0, total_add + 1, step_add))
#     add_percentages  = [100 * k / n for k in add_counts]
#
#     def _fit_eval(sub_idx):
#         """Train on sub_idx and return test accuracy."""
#         if len(sub_idx) == 0:
#             # No training data: predict majority class of test set as a baseline
#             majority = np.bincount(y_test).argmax()
#             return accuracy_score(y_test, np.full_like(y_test, majority))
#         y_sub = y_train[sub_idx]
#         # If single-class subset, use trivial classifier equal to that class
#         if len(np.unique(y_sub)) < 2:
#             return accuracy_score(y_test, np.full_like(y_test, y_sub[0]))
#         model = LogisticRegression(max_iter=5000, random_state=0)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model.fit(X_train[sub_idx], y_sub)
#         return accuracy_score(y_test, model.predict(X_test))
#
#     def _curve(order_idx):
#         """Build curve by adding points cumulatively following order_idx."""
#         accs = []
#         for k in add_counts:
#             keep = order_idx[:k]  # first k indices
#             accs.append(_fit_eval(keep))
#         return np.asarray(accs)
#
#     # Build curves for each valuation
#     results = {}
#     for lbl, val in zip(labels, val_arrays):
#         # Robust ranking with NaN handling:
#         if order == 'ascending':
#             # smallest first; NaNs to +inf so they go to the end (least informative)
#             rank_key = np.nan_to_num(val, nan=np.inf)
#             order_idx = np.argsort(rank_key)  # increasing
#         else:
#             # largest first; NaNs to -inf so they go to the end
#             rank_key = np.nan_to_num(val, nan=-np.inf)
#             order_idx = np.argsort(rank_key)[::-1]  # decreasing
#         results[lbl] = _curve(order_idx)
#
#     # Random baseline
#     random_curves = [_curve(np.random.permutation(n)) for _ in range(random_iters)]
#     results['Random'] = np.mean(random_curves, axis=0)
#
#     # Plot
#     plt.figure(figsize=(7, 5), dpi = 150)
#     for key, curve in results.items():
#         plt.plot(add_percentages, curve * 100, lw=3, marker='o', label=key)
#
#     title_dir = "High→Low" if order != 'ascending' else "Low→High"
#     # plt.title(f'Performance vs. Cumulative Addition ({title_dir})', fontsize=16)
#     plt.xlabel('Fraction of train data used (%)', fontsize = 50)
#     plt.ylabel('Accuracy (%)', fontsize = 50)
#     plt.grid(True)
#     plt.legend(fontsize = 50, loc='lower right')
#     if ylim is not None:
#         plt.ylim(*ylim)
#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.show()
#
#     return add_percentages, results

def PerformancePlot_Addition(vals, X_train, y_train, X_test, y_test,
                             a, b, *, order='descending', random_iters=10,
                             ylim=None, Names=None, save_path=None, random_state=0):
    """
    Addition plot: cumulatively add training points according to a ranking
    (descending = highest-valued first; ascending = lowest-valued first)
    and track test accuracy after each addition.

    Returns
    -------
    add_percentages : list[float]
    results : dict[str, np.ndarray]
    """
    # --- NEW: match rcParams with PerformancePlot_Removal ---
    plt.rcParams.update({
        "font.size": 28,        # base font size
        "axes.titlesize": 28,   # title font
        "axes.labelsize": 28,   # axis labels
        "xtick.labelsize": 28,  # tick labels
        "ytick.labelsize": 28,
        "legend.fontsize": 23,  # legend
    })

    # Parse inputs
    if isinstance(vals, Mapping):
        labels, val_arrays = list(vals.keys()), [np.asarray(v, dtype=float) for v in vals.values()]
    else:
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        val_arrays = [np.asarray(v, dtype=float) for v in vals]
        labels = (list(Names) if Names is not None else
                  [getattr(v, 'name', None) for v in val_arrays])
        labels = [lbl if lbl else f'valuation {i + 1}' for i, lbl in enumerate(labels)]

    n = X_train.shape[0]
    assert all(len(v) == n for v in val_arrays), "All valuation arrays must match |X_train|"

    # Step schedule
    total_add = int(np.floor(b * n))
    step_add  = max(1, int(np.floor(a * n)))
    add_counts      = list(range(0, total_add + 1, step_add))
    add_percentages = [100 * k / n for k in add_counts]

    def _fit_eval(sub_idx):
        if len(sub_idx) == 0:
            majority = np.bincount(y_test).argmax()
            return accuracy_score(y_test, np.full_like(y_test, majority))
        y_sub = y_train[sub_idx]
        if len(np.unique(y_sub)) < 2:
            return accuracy_score(y_test, np.full_like(y_test, y_sub[0]))
        model = LogisticRegression(max_iter=5000, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train[sub_idx], y_sub)
        return accuracy_score(y_test, model.predict(X_test))

    def _curve(order_idx):
        accs = []
        for k in add_counts:
            keep = order_idx[:k]
            accs.append(_fit_eval(keep))
        return np.asarray(accs)

    # Curves for each valuation
    results = {}
    for lbl, val in zip(labels, val_arrays):
        if order == 'ascending':
            rank_key = np.nan_to_num(val, nan=np.inf)
            order_idx = np.argsort(rank_key)
        else:
            rank_key = np.nan_to_num(val, nan=-np.inf)
            order_idx = np.argsort(rank_key)[::-1]
        results[lbl] = _curve(order_idx)

    # Random baseline
    rng = np.random.default_rng(random_state)
    random_curves = [_curve(rng.permutation(n)) for _ in range(random_iters)]
    results['Random'] = np.mean(random_curves, axis=0)

    # ---- Plot: aligned with PerformancePlot_Removal ----
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, layout='constrained')

    for key, curve in results.items():
        ax.plot(add_percentages, curve * 100, lw=5, marker='o', label=key)

    ax.set_xlabel('Fraction of train data used (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True)
    # match the first function's margins
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.98, top=0.90)

    if ylim is not None:
        plt.ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper', nbins=3))

    if save_path is not None:
        # match save behavior of PerformancePlot_Removal
        plt.savefig(save_path, format='pdf')

    plt.show()

    return add_percentages, results