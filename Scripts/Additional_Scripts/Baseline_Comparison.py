'''
0. Import Packages
'''
import time
import json
import os
import warnings
import numpy as np
import pandas as pd
import inspect
from math import ceil, sqrt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from numpy.typing import ArrayLike
from collections.abc import Mapping as MappingABC
from typing import Callable, Dict, Mapping, List, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pydvl.valuation import (Dataset, SupervisedScorer, ModelUtility, ShapleyValuation,
                             BetaShapleyValuation, DeltaShapleyValuation, GroupTestingShapleyValuation, ClasswiseShapleyValuation)
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility
from pydvl.valuation.samplers.classwise import ClasswiseSampler
from pydvl.valuation.samplers import PermutationSampler, OwenSampler, GridOwenStrategy
from pydvl.valuation.samplers.powerset import UniformSampler, NoIndexIteration
from pydvl.valuation.samplers.truncation import RelativeTruncation
from pydvl.valuation.stopping import (RankCorrelation, MinUpdates, MaxSamples, History, NoStopping)
from pydvl.valuation.methods.knn_shapley import KNNShapleyValuation
from pydvl.valuation.samplers.stratified import DeltaShapleyNCSGDSampleSize
from pydvl.valuation.samplers.stratified import StratifiedSampler, PowerLawSampleSize


'''
1. Load Data
'''
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/Vehicle_01_1"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/Vehicle_01_test_1"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/Vehicle_01_TMC_1"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle"
LABEL_COL = "target"
# PRED_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_22_Pred.csv"
runtimes = {}
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()
X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()
train_ds = Dataset(X_train, y_train)
test_ds  = Dataset(X_test,  y_test)

def timed_fit(name, val_obj, train_ds):
    t0 = time.perf_counter()
    val_obj.fit(train_ds)
    dt = time.perf_counter() - t0
    runtimes[name] = dt
    print(f"[runtime] {name}: {dt:.3f} s")
    return val_obj

# ---------- Base models ----------
# (A) Logistic regression for model-based methods
lr = LogisticRegression(max_iter=200)
scorer = SupervisedScorer("accuracy", test_ds, default=0.0, range=(0.0, 1.0))
utility = ModelUtility(lr, scorer)

# (B) KNN for KNN-Shapley
knn = KNeighborsClassifier(n_neighbors=5)

# ---------- Common stoppers ----------
shapley_stopping   = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
beta_stopping      = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
delta_stopping     = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
group_stopping     = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)


'''
2. Compute Valuations
'''
# ---------- 1) Permutation-based Shapley ----------
perm_sampler = PermutationSampler(truncation=RelativeTruncation(rtol=0.01, burn_in_fraction=0.3), seed=42)
val_shapley  = ShapleyValuation(utility, perm_sampler, shapley_stopping, progress=True)
timed_fit("PermutedShapley", val_shapley, train_ds)
res_shapley = val_shapley.result.values

# ---------- 2) Beta Shapley ----------
val_beta = BetaShapleyValuation(utility, perm_sampler, beta_stopping, alpha=16, beta=1, progress=True)
timed_fit("BetaShapley", val_beta, train_ds)
res_beta = val_beta.result.values

# ---------- 3) Class-wise Shapley ----------
cw_scorer  = ClasswiseSupervisedScorer("accuracy", test_ds, default=0.0, range=(0.0, 1.0))
cw_utility = ClasswiseModelUtility(lr, cw_scorer)
cw_sampler = ClasswiseSampler(
    in_class=PermutationSampler(seed=42),
    out_of_class=UniformSampler(index_iteration=NoIndexIteration),
    max_in_class_samples=1
)
n_labels = len(np.unique(y_train))
cw_stop  = MaxSamples(cw_sampler, 500 * n_labels) | History(n_steps=5000)
val_cs   = ClasswiseShapleyValuation(utility=cw_utility, sampler=cw_sampler, is_done=cw_stop, normalize_values=True)
timed_fit("ClasswiseShapley", val_cs, train_ds)
cs_values = val_cs.result.values

# ---------- 4) Delta Shapley ----------
n = len(train_ds)
l, u = n//3, (2*n)//3
size_strategy = PowerLawSampleSize(lower_bound=l, upper_bound=u, exponent=2.0, n_samples=1000)
sampler = StratifiedSampler(sample_sizes=size_strategy, seed=42)
val_delta = DeltaShapleyValuation(utility=utility, sampler=sampler, is_done=delta_stopping, progress=True)
timed_fit("DeltaShapley", val_delta, train_ds)
delta_vals = val_delta.result.values

# ---------- 5) Group Testing Shapley ----------
val_group = GroupTestingShapleyValuation(utility=utility, progress=True, n_samples=2000, epsilon=0.05)
timed_fit("GroupTesting", val_group, train_ds)
group_vals = val_group.result.values

# ---------- 6) Owen Shapley ----------
owen_sampler = OwenSampler(outer_sampling_strategy=GridOwenStrategy(n_samples_outer=200), n_samples_inner=2)
owen_stop = NoStopping(owen_sampler)
owen_val = ShapleyValuation(utility, owen_sampler, owen_stop, progress=True)

from joblib import parallel_config
t0 = time.perf_counter()
with parallel_config(n_jobs=4):
    owen_val.fit(train_ds)
runtimes["OwenShapley"] = time.perf_counter() - t0
print(f"[runtime] OwenShapley: {runtimes['OwenShapley']:.3f} s")

res_owen = owen_val.result.values

# ---------- 7) KNN Shapley ----------
knn_val = KNNShapleyValuation(knn, test_ds, progress=True)
timed_fit("KNNShapley", knn_val, train_ds)
knn_vals = knn_val.result.values

# --- persist runtimes (optional) ---
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame([runtimes]).T.rename(columns={0: "seconds"}).to_csv(os.path.join(OUT_DIR, "runtimes.csv"))
with open(os.path.join(OUT_DIR, "runtimes.json"), "w") as f:
    json.dump(runtimes, f, indent=2)

# ---------- Collect & save ----------
shapley_df = pd.DataFrame({
    # # "Permutated_Shapley":    res_shapley,
    "BetaShapley":            res_beta,
    "DeltaShapley":           delta_vals,
    "GroupTesting":           group_vals,
    "OwenShapley":            res_owen,
    "Classwise_Shapley":       cs_values,
    "KNNShapley":             knn_vals
})
# Comparison_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251111/FMNIST_0469_Comparison.csv"
pred_df = pd.read_csv(TRAIN_CSV)
# pred_df = pd.read_csv(Comparison_CSV)
pred_df = pred_df.reset_index(drop=True)
#df_train = df_train.reset_index(drop=True)
combined_df = pd.concat([pred_df, shapley_df], axis=1)
combined_df.to_csv(os.path.join(OUT_DIR, "characters_048_Comparison.csv"), index=False)


'''
3. Linear Comparison Plot
'''
# =========================
# 3.0 Load CSVs (as you do)
# =========================
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle/Vehicle_01_1"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle/Vehicle_01_test_1"
PRED_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle/Vehicle_01_1_Pred.csv"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle/Vehicle_01_TMC_1"
OUT_DIR2  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260105/baseline2/Vehicle"

df      = pd.read_csv(os.path.join(OUT_DIR2, "Vehicle_01_1_Comparison.csv"))
df_pred = pd.read_csv(PRED_CSV)
df_tmc  = pd.read_csv(TMC_CSV)

# =========================
# 3.1 Extract arrays
# =========================
y_vals    = df["target"].to_numpy()
per_vals  = df["PerDShap"].to_numpy()

beta_vals  = df["BetaShapley"].to_numpy()
delta_vals = df["DeltaShapley"].to_numpy()
gt_vals    = df["GroupTesting"].to_numpy()
owen_vals  = df["OwenShapley"].to_numpy()
cw_vals    = df["Classwise_Shapley"].to_numpy()
knn_vals   = df["KNNShapley"].to_numpy()

tmc_vals   = df_tmc["TMC_Shap"].to_numpy()
pred_vals  = df_pred["Pred_Shap"].to_numpy()

# =========================
# 3.2 Configs (colors / markers)
# =========================
# Two series will be overlayed in each paired plot:
#   Predicted Shapley (fixed)
#   Baseline method (one of 7)
COLORS = {
    "Predicted Shapley": "#1f77b4",  # blue
    "TMC Shapley":       "#ff7f0e",  # orange
    "Beta Shapley":      "#2ca02c",  # green
    "Delta Shapley":     "#d62728",  # red
    "Group Testing":     "#9467bd",  # purple
    "Owen Shapley":      "#8c564b",  # brown
    "Classwise Shapley": "#e377c2",  # pink
    "KNN Shapley":       "#17becf",  # cyan
}

classes = np.unique(y_vals)
MARKERS = ['o','s','^','D','P','X','v','<','>','*']
styles  = {
    c: dict(marker=MARKERS[i % len(MARKERS)], s=22, linewidths=1.2)
    for i, c in enumerate(classes)
}

def _get_bounds(x, ys):
    """Compute shared axis bounds over x and list of y arrays, ignoring NaNs."""
    vv = [pd.to_numeric(x, errors="coerce")]
    for y in ys:
        vv.append(pd.to_numeric(y, errors="coerce"))
    vv = np.concatenate(vv)
    vv = vv[~np.isnan(vv)]
    if vv.size == 0:
        return None
    lo, hi = float(np.min(vv)), float(np.max(vv))
    pad = 0.001 * (hi - lo + 1e-12)
    return lo, hi, pad

# Use PerDShap as the common x-axis for everything in this Part 3
x = pd.to_numeric(per_vals, errors="coerce")

# =========================
# 3.3 Build "all series" dict (for all-in-one plot)
# =========================
series_all = {
    "Predicted Shapley":  pred_vals,
    "TMC Shapley":        tmc_vals,
    "Beta Shapley":       beta_vals,
    "Delta Shapley":      delta_vals,
    "Group Testing":      gt_vals,
    "Owen Shapley":       owen_vals,
    "Classwise Shapley":  cw_vals,
    "KNN Shapley":        knn_vals,
}

# =========================
# 3.4 Plot (1) ALL-IN-ONE
# =========================
# valid for x only; each series will have its own NaN mask
all_y = np.concatenate([pd.to_numeric(v, errors="coerce") for v in series_all.values()])
valid_x = ~np.isnan(x)

ys_list = [pd.to_numeric(v, errors="coerce") for v in series_all.values()]
bounds = _get_bounds(x, ys_list)
if bounds is None:
    print("[WARN] all-in-one plot skipped: no valid points.")
else:
    lo, hi, pad = bounds
    fig, ax = plt.subplots(figsize=(9.2, 9.2))

    for c in classes:
        m_class = (y_vals == c) & valid_x
        st = styles[c].copy()

        for name, y in series_all.items():
            y_num = pd.to_numeric(y, errors="coerce")
            m = m_class & ~np.isnan(y_num)
            if not np.any(m):
                continue
            ax.scatter(
                x[m], y_num[m],
                marker=st["marker"], s=st["s"],
                facecolors=COLORS.get(name, "gray"), edgecolors="k", alpha=0.6
            )

    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--k", linewidth=1.4)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Permuted Shapley Value (PerDShap)")
    ax.set_ylabel("Shapley Variants (all)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # series legend
    series_handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none',
               markeredgecolor=COLORS.get(name, "gray"), markersize=8, label=name)
        for name in series_all.keys()
    ]
    legend_series = ax.legend(handles=series_handles, title="Series (color)",
                              loc="upper left", fontsize=9)
    ax.add_artist(legend_series)

    # class legend
    class_handles = [
        Line2D([0],[0], marker=styles[c]["marker"], linestyle="None",
               markerfacecolor="none", markeredgecolor="k",
               markersize=8, label=f"class {int(c)}")
        for c in classes
    ]
    legend_classes = ax.legend(handles=class_handles, title="Class (marker)",
                               loc="lower right", fontsize=9)
    ax.add_artist(legend_classes)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR2, "Vehicle_01_AllBaselinesPlusPred_vs_PerDShap.png")
    plt.savefig(out_png, dpi=500)
    plt.close(fig)
    print("Saved ALL-IN-ONE:", out_png)


# =========================
# 3.5 Plot (2) SEVEN PAIRED PLOTS
#     Each plot overlays:
#         (a) baseline vs PerDShap
#         (b) predicted vs PerDShap
# =========================
methods = {
    "TMC Shapley":       tmc_vals,
    "Beta Shapley":      beta_vals,
    "Delta Shapley":     delta_vals,
    "Group Testing":     gt_vals,
    "Owen Shapley":      owen_vals,
    "Classwise Shapley": cw_vals,
    "KNN Shapley":       knn_vals,
}

y_pred = pd.to_numeric(pred_vals, errors="coerce")

for bname, y_base_raw in methods.items():
    y_base = pd.to_numeric(y_base_raw, errors="coerce")

    valid = (~np.isnan(x)) & (~np.isnan(y_pred)) & (~np.isnan(y_base))
    if not np.any(valid):
        print(f"[WARN] Skip paired plot for {bname}: no valid points")
        continue

    bounds = _get_bounds(x[valid], [y_pred[valid], y_base[valid]])
    if bounds is None:
        print(f"[WARN] Skip paired plot for {bname}: cannot compute bounds")
        continue

    lo, hi, pad = bounds
    pad *= 2
    fig, ax = plt.subplots(figsize=(9.2, 9.2))

    for c in classes:
        m = valid & (y_vals == c)
        if not np.any(m):
            continue
        mk = styles[c]["marker"]
        ss = styles[c]["s"]

        # Predicted vs PerDShap (one color)
        ax.scatter(
            x[m], y_pred[m],
            marker=mk, s=ss,
            facecolors="none", edgecolors=COLORS["Predicted Shapley"],
            linewidths=1.2, alpha=0.9
        )

        # Baseline vs PerDShap (baseline color)
        ax.scatter(
            x[m], y_base[m],
            marker=mk, s=ss,
            facecolors="none", edgecolors=COLORS.get(bname, "gray"),
            linewidths=1.2, alpha=0.9
        )

    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--k", linewidth=1.4)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Permutated-based Data Shapley Value (PerDShap)")
    ax.set_ylabel("Shapley Value")
    ax.set_title(f"{bname} + Predicted vs PerDShap")
    ax.grid(True, linestyle="--", alpha=0.5)

    # series legend (two entries)
    series_handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none',
               markeredgecolor=COLORS["Predicted Shapley"], markersize=8, label="Predicted Shapley"),
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none',
               markeredgecolor=COLORS.get(bname, "gray"), markersize=8, label=bname),
    ]
    legend_series = ax.legend(handles=series_handles, title="Series (color)",
                              loc="upper left", fontsize=9)
    ax.add_artist(legend_series)

    # class legend
    class_handles = [
        Line2D([0],[0], marker=styles[c]["marker"], linestyle="None",
               markerfacecolor="none", markeredgecolor="k",
               markersize=8, label=f"class {int(c)}")
        for c in classes
    ]
    ax.legend(handles=class_handles, title="Class (marker)",
              loc="lower right", fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(
        OUT_DIR2,
        f"Vehicle_01_1_{bname.replace(' ','').replace('-','')}_AND_Pred_vs_Per.png"
    )
    plt.savefig(out_png, dpi=500)
    plt.close(fig)
    print("Saved PAIRED:", out_png)


'''
4. MSE Computation
'''
truth = per_vals
variants = {
    "Predicted Shapley":  pred_vals,
    "TMC Shapley":         tmc_vals,
    "Beta Shapley":        beta_vals,
    "Delta Shapley":       delta_vals,
    "Group Testing":       gt_vals,
    "Owen Shapley":        owen_vals,
    "Classwise Shapley":   cw_vals,
    "KNN Shapley":         knn_vals
}
mse_results = {}
for name, vals in variants.items():
    mask = (~np.isnan(truth)) & (~np.isnan(vals))
    if mask.sum() == 0:
        mse = np.nan
    else:
        diff = truth[mask] - vals[mask]
        mse = np.mean(np.square(diff))
    mse_results[name] = mse

print("MSE vs PerDShap (ground truth):")
for name, mse in sorted(mse_results.items(), key=lambda x: x[1]):
    print(f"{name:<20s} MSE = {mse:.6g}")

print("\nRMSE vs PerDShap:")
for name, mse in sorted(mse_results.items(), key=lambda x: x[1]):
    rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan
    print(f"{name:<20s} RMSE = {rmse:.6g}")

'''
5. Removal and Addition Plots
'''
df = pd.read_csv(TRAIN_CSV)
df_tmc = pd.read_csv(TMC_CSV)
y_vals = df["target"].to_numpy()
per_vals = df["PerDShap"].to_numpy()

def _as_1d(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def _check_same_length(*arrays: ArrayLike) -> None:
    lengths = {len(np.asarray(a)) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Arrays must have same length, got lengths {lengths}")


def _default_model_builder():
    """Default downstream model for utility."""
    return LogisticRegression(max_iter=5000, multi_class="auto", n_jobs=-1, random_state=0)


def _load_numpy_dataset(name: str = "vehicle") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supported datasets:
        - 'Vehicle': OpenML id 54
        - 'MNIST': TF datasets
        - 'FashionMNIST': TF datasets
        - 'CIFAR10': TF datasets
        - 'CIFAR100': TF datasets
    """
    name_l = name.lower()

    if name_l in {"mnist", "fashion_mnist", "fashionmnist"}:
        import tensorflow as tf
        if name_l.startswith("fashion"):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
        X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
        return X_train, y_train, X_test, y_test

    if name_l in {"cifar10", "cifar-10"}:
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
        X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
        return X_train, y_train, X_test, y_test

    if name_l in {"cifar100", "cifar-100"}:
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
        X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
        return X_train, y_train, X_test, y_test

    if name_l in {"vehicle", "vehicles"}:
        from sklearn.datasets import fetch_openml

        data = fetch_openml("vehicle", version=1, as_frame=False)
        X = data.data.astype("float32")

        y_str = np.asarray(data.target)
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        return X_train, y_train, X_test, y_test

    raise ValueError(
        f"Unknown dataset name '{name}'. Supported: "
        "Vehicle, MNIST, Fashion_MNIST, CIFAR10, CIFAR100."
    )


def _prepare_dataset(*, dataset: Optional[str] = "vehicle", random_state: int = 0,
                     X_train: Optional[ArrayLike] = None, y_train: Optional[ArrayLike] = None,
                     X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                     n_train: Optional[int] = None, n_test: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two modes:
    1) Built-in dataset: dataset != None
    2) User provided: dataset=None and X_train/y_train/X_test/y_test all given
    """
    if dataset is not None:
        if any(v is not None for v in (X_train, y_train, X_test, y_test)):
            raise ValueError("Use `dataset` OR provide explicit X_train/y_train/X_test/y_test, not both.")
        X_train, y_train, X_test, y_test = _load_numpy_dataset(dataset)
    else:
        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("If dataset=None, you must provide X_train,y_train,X_test,y_test.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    rng = np.random.default_rng(random_state)

    if n_train is not None and n_train < len(X_train):
        idx = rng.permutation(len(X_train))[:n_train]
        X_train = X_train[idx]
        y_train = y_train[idx]

    if n_test is not None and n_test < len(X_test):
        idx = rng.permutation(len(X_test))[:n_test]
        X_test = X_test[idx]
        y_test = y_test[idx]

    return X_train, y_train, X_test, y_test

def _removal_curve_from_values(values: ArrayLike, *,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               order: str, step: float, max_fraction: float,
                               model_builder: Callable[[], object]) -> Tuple[np.ndarray, np.ndarray]:
    vals = _as_1d(values)
    n = len(vals)
    if n != len(X_train):
        raise ValueError(f"values length {n}, but X_train length {len(X_train)}")

    if not (0 < step <= 1):
        raise ValueError("step must be in (0,1].")
    if not (0 < max_fraction <= 1):
        raise ValueError("max_fraction must be in (0,1].")

    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]
    elif order != "ascending":
        raise ValueError("order must be 'descending' or 'ascending'.")

    fractions = np.arange(0.0, max_fraction + 1e-12, step)
    all_idx = np.arange(n)
    utilities: List[float] = []

    for frac in fractions:
        k = int(round(frac * n))
        k = min(max(k, 0), n)

        to_remove = order_idx[:k]
        mask = np.ones(n, dtype=bool)
        mask[to_remove] = False
        keep = all_idx[mask]

        if len(keep) == 0:
            utilities.append(0.0)
            continue

        X_sub = X_train[keep]
        y_sub = y_train[keep]
        if len(np.unique(y_sub)) < 2:
            majority = np.bincount(y_test).argmax()
            utilities.append(accuracy_score(y_test, np.full_like(y_test, majority)))
            continue

        model = model_builder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sub, y_sub)
        utilities.append(accuracy_score(y_test, model.predict(X_test)))

    return fractions, np.asarray(utilities, dtype=float)


def _addition_curve_from_values(values: ArrayLike, *,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                order: str, step: float, max_fraction: float,
                                model_builder: Callable[[], object]) -> Tuple[np.ndarray, np.ndarray]:
    vals = _as_1d(values)
    n = len(vals)
    if n != len(X_train):
        raise ValueError(f"values length {n}, but X_train length {len(X_train)}")

    if not (0 < step <= 1):
        raise ValueError("step must be in (0,1].")
    if not (0 < max_fraction <= 1):
        raise ValueError("max_fraction must be in (0,1].")

    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]
    elif order != "ascending":
        raise ValueError("order must be 'descending' or 'ascending'.")

    fractions = np.arange(0.0, max_fraction + 1e-12, step)
    utilities: List[float] = []

    for frac in fractions:
        k = int(round(frac * n))
        k = min(max(k, 1), n)
        subset = order_idx[:k]

        X_sub = X_train[subset]
        y_sub = y_train[subset]
        if len(np.unique(y_sub)) < 2:
            majority = np.bincount(y_test).argmax()
            utilities.append(accuracy_score(y_test, np.full_like(y_test, majority)))
            continue

        model = model_builder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sub, y_sub)
        utilities.append(accuracy_score(y_test, model.predict(X_test)))

    return fractions, np.asarray(utilities, dtype=float)


def removal_plot_all_methods(vals: Mapping[str, ArrayLike], *,
                             dataset: Optional[str] = "vehicle",
                             X_train: Optional[ArrayLike] = None, y_train: Optional[ArrayLike] = None,
                             X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                             n_train: Optional[int] = None, n_test: Optional[int] = None,
                             random_state: int = 0,
                             order: str = "descending",
                             step: float = 0.05, max_fraction: float = 0.4,
                             random_iters: int = 10,
                             model_builder: Optional[Callable[[], object]] = None,
                             ylim: Optional[Tuple[float, float]] = None,
                             title: str = "Data removal performance (A1)",
                             save_path: Optional[str] = None,
                             show: bool = False
                             ) -> Dict[str, np.ndarray]:
    """
    One figure with ALL methods (e.g., 9 lines incl. PerDShap) + Random baseline (dashed).
    """
    if not isinstance(vals, MappingABC):
        raise ValueError("vals must be a dict-like mapping: {name: values}.")

    X_tr, y_tr, X_te, y_te = _prepare_dataset(
        dataset=dataset, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_train=n_train, n_test=n_test, random_state=random_state
    )
    if model_builder is None:
        model_builder = _default_model_builder

    # Ensure all valuation vectors align with this X_tr
    for k, v in vals.items():
        v1 = _as_1d(v)
        if len(v1) != len(X_tr):
            raise ValueError(f"Method {k!r} has length {len(v1)} but X_train has length {len(X_tr)}.")

    # Compute curves
    fractions = None
    curves: Dict[str, np.ndarray] = {}
    for name, v in vals.items():
        f, u = _removal_curve_from_values(
            v, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            order=order, step=step, max_fraction=max_fraction,
            model_builder=model_builder
        )
        if fractions is None:
            fractions = f
        else:
            if not np.allclose(fractions, f):
                raise RuntimeError("Fractions mismatch between methods.")
        curves[name] = u

    assert fractions is not None

    # Random baseline (average over random permutations)
    rng = np.random.default_rng(random_state)
    n = len(y_tr)
    all_idx = np.arange(n)

    random_utils = np.zeros((random_iters, len(fractions)), dtype=float)
    for it in range(random_iters):
        perm = rng.permutation(n)
        u_list: List[float] = []
        for frac in fractions:
            k = int(round(frac * n))
            k = min(max(k, 0), n)
            to_remove = perm[:k]

            mask = np.ones(n, dtype=bool)
            mask[to_remove] = False
            keep = all_idx[mask]

            if len(keep) == 0:
                u_list.append(0.0)
                continue

            X_sub = X_tr[keep]
            y_sub = y_tr[keep]
            if len(np.unique(y_sub)) < 2:
                majority = np.bincount(y_te).argmax()
                u_list.append(accuracy_score(y_te, np.full_like(y_te, majority)))
                continue

            model = model_builder()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            u_list.append(accuracy_score(y_te, model.predict(X_te)))

        random_utils[it, :] = np.asarray(u_list, dtype=float)

    random_curve = random_utils.mean(axis=0)
    curves["Random"] = random_curve

    # Plot
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=300)
    x = fractions * 100.0

    # plot all methods (solid)
    for name, curve in vals.items():
        ax.plot(x, curves[name] * 100.0, marker="o", linewidth=2.0, label=name)

    # random dashed
    ax.plot(x, random_curve * 100.0, linestyle="--", linewidth=2.0, label="Random")

    ax.set_xlabel("Percentage of training data removed (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=9, ncol=1)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return curves


def addition_plot_all_methods(vals: Mapping[str, ArrayLike], *,
                              dataset: Optional[str] = "vehicle",
                              X_train: Optional[ArrayLike] = None, y_train: Optional[ArrayLike] = None,
                              X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                              n_train: Optional[int] = None, n_test: Optional[int] = None,
                              random_state: int = 0,
                              order: str = "descending",
                              step: float = 0.05, max_fraction: float = 1.0,
                              random_iters: int = 10,
                              model_builder: Optional[Callable[[], object]] = None,
                              ylim: Optional[Tuple[float, float]] = None,
                              title: str = "Data addition performance (A1)",
                              save_path: Optional[str] = None,
                              show: bool = False
                              ) -> Dict[str, np.ndarray]:
    """
    One figure with ALL methods (e.g., 9 lines incl. PerDShap) + Random baseline (dashed).
    """
    if not isinstance(vals, MappingABC):
        raise ValueError("vals must be a dict-like mapping: {name: values}.")

    X_tr, y_tr, X_te, y_te = _prepare_dataset(
        dataset=dataset, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_train=n_train, n_test=n_test, random_state=random_state
    )
    if model_builder is None:
        model_builder = _default_model_builder

    for k, v in vals.items():
        v1 = _as_1d(v)
        if len(v1) != len(X_tr):
            raise ValueError(f"Method {k!r} has length {len(v1)} but X_train has length {len(X_tr)}.")

    fractions = None
    curves: Dict[str, np.ndarray] = {}
    for name, v in vals.items():
        f, u = _addition_curve_from_values(
            v, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            order=order, step=step, max_fraction=max_fraction,
            model_builder=model_builder
        )
        if fractions is None:
            fractions = f
        else:
            if not np.allclose(fractions, f):
                raise RuntimeError("Fractions mismatch between methods.")
        curves[name] = u

    assert fractions is not None

    # Random baseline
    rng = np.random.default_rng(random_state)
    n = len(y_tr)

    random_utils = np.zeros((random_iters, len(fractions)), dtype=float)
    for it in range(random_iters):
        perm = rng.permutation(n)
        u_list: List[float] = []
        for frac in fractions:
            k = int(round(frac * n))
            k = min(max(k, 1), n)
            subset = perm[:k]

            X_sub = X_tr[subset]
            y_sub = y_tr[subset]
            if len(np.unique(y_sub)) < 2:
                majority = np.bincount(y_te).argmax()
                u_list.append(accuracy_score(y_te, np.full_like(y_te, majority)))
                continue

            model = model_builder()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            u_list.append(accuracy_score(y_te, model.predict(X_te)))

        random_utils[it, :] = np.asarray(u_list, dtype=float)

    random_curve = random_utils.mean(axis=0)
    curves["Random"] = random_curve

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=300)
    x = fractions * 100.0

    for name in vals.keys():
        ax.plot(x, curves[name] * 100.0, marker="o", linewidth=2.0, label=name)
    ax.plot(x, random_curve * 100.0, linestyle="--", linewidth=2.0, label="Random")

    ax.set_xlabel("Percentage of training data used (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=9, ncol=1)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return curves

def noisy_label_detection_curve(values: ArrayLike, is_noisy: ArrayLike, *,
                                order: str = "ascending") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (fraction_inspected, recall)
      - fraction_inspected: from 1/n ... 1
      - recall: fraction of noisy points found among inspected points
    """
    vals = _as_1d(values)
    noisy = np.asarray(is_noisy).astype(bool)
    if len(vals) != len(noisy):
        raise ValueError("values and is_noisy must have same length.")

    n_noisy = int(noisy.sum())
    if n_noisy == 0:
        raise ValueError("No noisy labels found in is_noisy; A2 undefined.")

    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]
    elif order != "ascending":
        raise ValueError("order must be 'ascending' or 'descending'.")

    cum_noisy_found = np.cumsum(noisy[order_idx].astype(int))
    inspected = np.arange(1, len(vals) + 1)
    recall = cum_noisy_found / n_noisy
    frac_inspected = inspected / len(vals)
    return frac_inspected, recall


def noisy_label_detection_plot_all_methods(vals: Mapping[str, ArrayLike], *,
                                           is_noisy: Optional[ArrayLike] = None,
                                           order: str = "ascending",
                                           title: str = "Noisy-label detection (A2)",
                                           save_path: Optional[str] = None,
                                           add_random: bool = True,
                                           synthetic_noise: bool = True,
                                           noise_fraction: float = 0.2,
                                           random_state: int = 0,
                                           show: bool = False
                                           ) -> Dict[str, Dict[str, float]]:
    """
    One figure with ALL methods (e.g., 9 lines) + Random baseline (dashed).
    Returns a summary dict including AUC(recall vs fraction), ROC-AUC, AP.
    """
    if not isinstance(vals, MappingABC):
        raise ValueError("vals must be a dict-like mapping: {name: values}.")

    # infer n from first method
    labels = list(vals.keys())
    val_arrays = [np.asarray(v) for v in vals.values()]
    n = len(val_arrays[0])

    # sanity: equal lengths
    for name, v in zip(labels, val_arrays):
        v1 = _as_1d(v)
        if len(v1) != n:
            raise ValueError(f"Method {name!r} has length {len(v1)} but expected {n}.")

    # is_noisy mask
    if is_noisy is None:
        if not synthetic_noise:
            raise ValueError("Provide is_noisy when synthetic_noise=False.")
        rng = np.random.default_rng(random_state)
        is_noisy = np.zeros(n, dtype=bool)
        k = max(1, int(round(noise_fraction * n)))
        idx = rng.choice(n, size=k, replace=False)
        is_noisy[idx] = True
    else:
        is_noisy = np.asarray(is_noisy).astype(bool)
        if len(is_noisy) != n:
            raise ValueError(f"is_noisy length {len(is_noisy)} but expected {n}.")

    summary: Dict[str, Dict[str, float]] = {}

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=300)

    y_true = is_noisy.astype(int)
    for name, v in zip(labels, val_arrays):
        frac, recall = noisy_label_detection_curve(v, is_noisy, order=order)
        ax.plot(frac * 100, recall * 100, marker="o", linewidth=2.0, label=name)

        # area under recall-vs-fraction curve
        area = float(np.trapz(recall, frac))

        # treat "scores" as a detector for noisy: smaller values mean more suspicious if order="ascending"
        scores = (-v) if order == "ascending" else v
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
            ap = float(average_precision_score(y_true, scores))
        except ValueError:
            roc_auc, ap = np.nan, np.nan

        summary[name] = {
            "AUC_recall_vs_fraction": area,
            "ROC_AUC": roc_auc,
            "AP": ap,
        }

    # Random baseline (dashed)
    if add_random:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        noisy_perm = is_noisy[perm]

        cum = np.cumsum(noisy_perm.astype(int))
        frac_r = np.arange(1, n + 1) / n
        recall_r = cum / max(1, is_noisy.sum())

        ax.plot(frac_r * 100, recall_r * 100, linestyle="--", linewidth=2.0, label="Random")

        area_r = float(np.trapz(recall_r, frac_r))
        rand_scores = rng.standard_normal(n)
        try:
            roc_auc_r = float(roc_auc_score(y_true, rand_scores))
            ap_r = float(average_precision_score(y_true, rand_scores))
        except ValueError:
            roc_auc_r, ap_r = np.nan, np.nan

        summary["Random"] = {
            "AUC_recall_vs_fraction": area_r,
            "ROC_AUC": roc_auc_r,
            "AP": ap_r,
        }

    ax.set_xlabel("Percentage of training points inspected (%)")
    ax.set_ylabel("Noisy-label recall (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=9, ncol=1)
    ax.set_xlim(0, 50)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return summary


# ============================================================
# 4. Example usage (you can delete this block)
# ============================================================
if __name__ == "__main__":
    # Suppose you have 9 valuation methods including Permutated-based DShap
    # Here we only create fake arrays to demonstrate API.
    rng = np.random.default_rng(0)
    n_points = len(X_train)

    methods_vals = {
        "Permutated-based DShap": rng.normal(size=n_points),
        "TMC Shapley": rng.normal(size=n_points),
        "Beta Shapley": rng.normal(size=n_points),
        "Delta Shapley": rng.normal(size=n_points),
        "Group Testing": rng.normal(size=n_points),
        "Owen Shapley": rng.normal(size=n_points),
        "Classwise Shapley": rng.normal(size=n_points),
        "KNN Shapley": rng.normal(size=n_points),
        "Predicted Shapley": rng.normal(size=n_points),
    }
    out_dir = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260104/baseline2/Vehicle"
    os.makedirs(out_dir, exist_ok=True)

    # A1 removal (if you have your own X_train/X_test, pass dataset=None and provide arrays)
    Xtr = X_train
    ytr = y_train
    Xte = X_test
    yte = y_test

    # _ = removal_plot_all_methods(
    #     methods_vals,
    #     dataset=None, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
    #     order="descending", step=0.05, max_fraction=0.4, random_iters=10,
    #     title="Removal Plot (High values first)",
    #     save_path=os.path.join(out_dir, "A1_Removal_AllMethods_High.png")
    # )
    # _ = removal_plot_all_methods(
    #     methods_vals,
    #     dataset=None, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
    #     order="ascending", step=0.05, max_fraction=0.4, random_iters=10,
    #     title="Removal Plot (Low values first)",
    #     save_path=os.path.join(out_dir, "A1_Removal_AllMethods_Low.png")
    # )

    # A2 noisy-label detection (9 methods in one plot)
    # Provide is_noisy if you have real noisy-label indices; otherwise synthetic.
    summary = noisy_label_detection_plot_all_methods(
        methods_vals,
        is_noisy=None, synthetic_noise=True, noise_fraction=0.1,
        order="ascending",
        title="Noisy-label detection",
        save_path=os.path.join(out_dir, "NoisyDetection_AllMethods.png")
    )

    print("A2 summary:", summary)