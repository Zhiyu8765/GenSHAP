# -*- coding: utf-8 -*-
"""
Models_Comparison.py

Task 1) Removal + Addition plots (4 plots per dataset base):
    - Removal (descending / ascending)
    - Addition (descending / ascending)
Each plot includes 6 lines:
    Pred_A, Pred_B, Pred_C, Pred_D, PerDShap, Random (average over RANDOM_ITERS)

Task 2) Noisy label detection:
    - Flip FLIP_FRAC of labels in y_train (multiclass-safe)
    - For each method, run KMeans(k=2) on 1D values
    - Mark noisy if value < min(cluster_centers)
    - Report F1 wrt flipped_mask
    - Save per-dataset results to CSV

This script is self-contained and does NOT depend on PerformancePlots_4lines.py.
"""

import os
import glob
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# =========================
# SETTINGS (edit if needed)
# =========================
COMPARISON_DIR = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260202/Comparison"
OUT_DIR        = os.path.join(COMPARISON_DIR, "Plots")

A_STEP       = 0.05   # step fraction
B_TOTAL      = 0.40   # total fraction to remove/add
RANDOM_ITERS = 10

FLIP_FRAC    = 0.10   # 10% label flips
SEED         = 0
TEST_SIZE    = 0.30   # proxy split inside each file

HOLDOUT_BASES = [
    "characters_048_1",
    "Iris_1",
    "STL10_3568x_3",
    "Vehicle_01_1",
    "volkert_15_2",
    "volkert_1245_1",
]
VARIANTS = ["A", "B", "C", "D"]

SUPPORTED_EXTS = (".csv", ".parquet", ".pkl", ".pickle")


# =========================
# IO + schema helpers
# =========================
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported extension: {ext} for {path}")


def find_file(comparison_dir: str, base: str, variant: str) -> str:
    pattern = os.path.join(comparison_dir, f"{base}_Pred_{variant}*")
    candidates = [
        p for p in glob.glob(pattern)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No file found for pattern {pattern} with exts {SUPPORTED_EXTS}"
        )
    return sorted(candidates)[0]


def infer_feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    exclude_set = set(exclude)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in exclude_set]
    if len(feat_cols) == 0:
        raise ValueError(
            "No numeric feature columns found after exclusion. "
            "Edit exclude_cols to match your dataframe schema."
        )
    return feat_cols


# =========================
# Core utilities: fitting
# =========================
def fit_and_score_lr(X_tr, y_tr, X_te, y_te) -> float:
    """
    Train a multiclass LogisticRegression and return test accuracy.
    Handles degenerate y_tr with <2 classes by returning dummy accuracy.
    """
    uniq = np.unique(y_tr)
    if len(uniq) < 2:
        dummy = np.full_like(y_te, uniq[0])
        return float(accuracy_score(y_te, dummy))

    # Multiclass-safe LR
    model = LogisticRegression(max_iter=400, random_state=0, n_jobs=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return float(accuracy_score(y_te, pred))


# =========================
# Removal / Addition curves
# =========================
def curve_removal(vals: np.ndarray,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  a: float,
                  b: float,
                  order: str) -> Tuple[List[float], np.ndarray]:
    """
    Removal curve:
        - order='descending': remove largest values first
        - order='ascending' : remove smallest values first
    """
    n = len(X_train)
    total_rm = int(np.floor(b * n))
    step_rm  = max(1, int(np.floor(a * n)))
    rm_counts = list(range(0, total_rm + 1, step_rm))
    rm_percentages = [100 * r / n for r in rm_counts]

    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]

    curve = []
    for r in rm_counts:
        if r == 0:
            acc = fit_and_score_lr(X_train, y_train, X_test, y_test)
            curve.append(acc)
            continue

        keep = order_idx[r:]  # remove first r points
        X_sub, y_sub = X_train[keep], y_train[keep]
        acc = fit_and_score_lr(X_sub, y_sub, X_test, y_test)
        curve.append(acc)

    return rm_percentages, np.asarray(curve)


def curve_addition(vals: np.ndarray,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   a: float,
                   b: float,
                   order: str) -> Tuple[List[float], np.ndarray]:
    """
    Addition curve:
        - order='descending': add largest values first
        - order='ascending' : add smallest values first
    """
    n = len(X_train)
    total_add = int(np.floor(b * n))
    step_add  = max(1, int(np.floor(a * n)))
    add_counts = list(range(0, total_add + 1, step_add))
    add_percentages = [100 * r / n for r in add_counts]

    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]

    curve = []
    for r in add_counts:
        if r == 0:
            # empty: majority baseline
            maj = int(pd.Series(y_train).mode().iloc[0])
            dummy = np.full_like(y_test, maj)
            curve.append(float(accuracy_score(y_test, dummy)))
            continue

        use = order_idx[:r]
        X_sub, y_sub = X_train[use], y_train[use]
        acc = fit_and_score_lr(X_sub, y_sub, X_test, y_test)
        curve.append(acc)

    return add_percentages, np.asarray(curve)


def random_baseline_removal(X_train, y_train, X_test, y_test, a, b, iters=10):
    n = len(X_train)
    total_rm = int(np.floor(b * n))
    step_rm  = max(1, int(np.floor(a * n)))
    rm_counts = list(range(0, total_rm + 1, step_rm))
    rm_percentages = [100 * r / n for r in rm_counts]

    curves = []
    for _ in range(iters):
        perm = np.random.permutation(n)
        curve = []
        for r in rm_counts:
            keep = perm[r:]
            acc = fit_and_score_lr(X_train[keep], y_train[keep], X_test, y_test)
            curve.append(acc)
        curves.append(curve)

    return rm_percentages, np.mean(np.asarray(curves), axis=0)


def random_baseline_addition(X_train, y_train, X_test, y_test, a, b, iters=10):
    n = len(X_train)
    total_add = int(np.floor(b * n))
    step_add  = max(1, int(np.floor(a * n)))
    add_counts = list(range(0, total_add + 1, step_add))
    add_percentages = [100 * r / n for r in add_counts]

    curves = []
    for _ in range(iters):
        perm = np.random.permutation(n)
        curve = []
        for r in add_counts:
            if r == 0:
                maj = int(pd.Series(y_train).mode().iloc[0])
                dummy = np.full_like(y_test, maj)
                curve.append(float(accuracy_score(y_test, dummy)))
            else:
                use = perm[:r]
                acc = fit_and_score_lr(X_train[use], y_train[use], X_test, y_test)
                curve.append(acc)
        curves.append(curve)

    return add_percentages, np.mean(np.asarray(curves), axis=0)


# =========================
# Plotting
# =========================
def plot_curves(x_pct: List[float],
                curves: Dict[str, np.ndarray],
                title: str,
                xlabel: str,
                ylabel: str,
                save_path: str):
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for name, c in curves.items():
        ax.plot(x_pct, c * 100.0, lw=4, marker="o", label=name)

    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.close(fig)


# =========================
# Noisy label detection
# =========================
def flip_labels(y: np.ndarray, flip_frac: float, seed: int):
    """
    Multiclass-safe flipping:
    For each selected index, replace label with a different class uniformly.
    Returns: y_flipped, flipped_mask
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    n = len(y)
    k = int(np.floor(flip_frac * n))
    idx = rng.choice(n, size=k, replace=False)

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes to flip labels, got classes={classes}")

    alt = {c: classes[classes != c] for c in classes}
    for i in idx:
        y[i] = rng.choice(alt[y[i]])

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return y, mask


def kmeans_noisy_detector(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    KMeans(k=2) on 1D values, noisy if value < min(cluster_centers).
    """
    v = np.asarray(values).reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    km.fit(v)
    centers = km.cluster_centers_.reshape(-1)
    thr = centers.min()
    return (v.reshape(-1) < thr)


# =========================
# Main
# =========================
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    noisy_rows = []

    for base in HOLDOUT_BASES:
        # ---- load A/B/C/D ----
        dfs = {}
        for v in VARIANTS:
            fpath = find_file(COMPARISON_DIR, base, v)
            dfs[v] = load_table(fpath)

        # checks
        for v in VARIANTS:
            if "Pred_Shap" not in dfs[v].columns:
                raise KeyError(f"[{base} Pred_{v}] missing 'Pred_Shap'")
            if "target" not in dfs[v].columns:
                raise KeyError(f"[{base} Pred_{v}] missing 'target'")

        df0 = dfs["A"].copy()
        if "PerDShap" not in df0.columns:
            raise KeyError(f"[{base}] missing 'PerDShap' in Pred_A (reference) file")

        n0 = len(df0)
        for v in VARIANTS[1:]:
            if len(dfs[v]) != n0:
                raise ValueError(f"[{base}] row mismatch: Pred_A={n0}, Pred_{v}={len(dfs[v])}")

        # ---- features ----
        exclude_cols = ["target", "Pred_Shap", "PerDShap"]
        feat_cols = infer_feature_columns(df0, exclude=exclude_cols)

        X = df0[feat_cols].to_numpy()
        y = df0["target"].to_numpy()

        # proxy split (within-file)
        idx = np.arange(len(df0))
        idx_train, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        X_train, y_train = X[idx_train], y[idx_train]
        X_test,  y_test  = X[idx_test],  y[idx_test]

        # values aligned to idx_train
        vals_map = {
            "Pred_A": dfs["A"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_B": dfs["B"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_C": dfs["C"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_D": dfs["D"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "PerDShap": df0.loc[idx_train, "PerDShap"].to_numpy(),
        }

        # ================
        # Task 1: Removal
        # ================
        for order in ["descending", "ascending"]:
            curves = {}
            x_pct_ref = None
            for name, v in vals_map.items():
                x_pct, c = curve_removal(v, X_train, y_train, X_test, y_test, A_STEP, B_TOTAL, order=order)
                curves[name] = c
                if x_pct_ref is None:
                    x_pct_ref = x_pct

            x_pct_r, c_r = random_baseline_removal(X_train, y_train, X_test, y_test, A_STEP, B_TOTAL, iters=RANDOM_ITERS)
            curves["Random"] = c_r

            title = f"{base} | Removal ({order})"
            save_path = os.path.join(OUT_DIR, f"{base}_Removal_{order}.png")
            plot_curves(
                x_pct_ref,
                curves,
                title=title,
                xlabel="Fraction of data removed (%)",
                ylabel="Accuracy (%)",
                save_path=save_path
            )

        # =================
        # Task 1: Addition
        # =================
        for order in ["descending", "ascending"]:
            curves = {}
            x_pct_ref = None
            for name, v in vals_map.items():
                x_pct, c = curve_addition(v, X_train, y_train, X_test, y_test, A_STEP, B_TOTAL, order=order)
                curves[name] = c
                if x_pct_ref is None:
                    x_pct_ref = x_pct

            x_pct_r, c_r = random_baseline_addition(X_train, y_train, X_test, y_test, A_STEP, B_TOTAL, iters=RANDOM_ITERS)
            curves["Random"] = c_r

            title = f"{base} | Addition ({order})"
            save_path = os.path.join(OUT_DIR, f"{base}_Addition_{order}.png")
            plot_curves(
                x_pct_ref,
                curves,
                title=title,
                xlabel="Fraction of data added (%)",
                ylabel="Accuracy (%)",
                save_path=save_path
            )

        # ==========================
        # Task 2: Noisy label detect
        # ==========================
        y_train_clean = y_train.copy()
        _, flipped_mask = flip_labels(y_train_clean, flip_frac=FLIP_FRAC, seed=SEED)

        for method_name, v in vals_map.items():
            pred_noisy = kmeans_noisy_detector(v, seed=SEED)
            f1 = f1_score(flipped_mask.astype(int), pred_noisy.astype(int), zero_division=0)
            noisy_rows.append({
                "holdout_base": base,
                "method": method_name,
                "flip_frac": FLIP_FRAC,
                "f1": float(f1),
                "n_train": int(len(y_train_clean)),
                "n_flipped": int(flipped_mask.sum()),
                "n_pred_noisy": int(pred_noisy.sum()),
            })

        print(f"[DONE] {base} plots + noisy detection finished.")

    # Save noisy label summary
    noisy_df = pd.DataFrame(noisy_rows)
    noisy_csv = os.path.join(OUT_DIR, "NoisyLabelDetection_F1.csv")
    noisy_df.to_csv(noisy_csv, index=False)

    summary = (noisy_df.groupby("method")["f1"]
               .agg(["mean", "std", "count"])
               .reset_index()
               .sort_values("mean", ascending=False))

    print("\n=== Noisy Label Detection (F1) Summary across holdouts ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-holdout F1 to: {noisy_csv}")
    print(f"Plots saved under: {OUT_DIR}")


if __name__ == "__main__":
    run()