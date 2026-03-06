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
from sklearn.base import clone
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
from pydvl.valuation.types import Sample
from pydvl.valuation.methods.banzhaf import BanzhafValuation
from pydvl.valuation.methods.semivalue import SemivalueValuation
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


' For Calculation of TMC Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
# os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
os.chdir('/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/DataShapley')
sys.path.append(os.getcwd())
#sys.path.append('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')

### Package and Environments for TMC_Shapley
#%load_ext autoreload
#%autoreload 2
#import shap_utils
#importlib.reload(shap_utils)
import numpy as np
import pandas as pd
from DShap import DShap
from shap_utils import *
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

#%matplotlib inline
MEM_DIR = '../'


'''
1. Load Data
'''
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris/Iris_1"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris/Iris_test_1"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris/Iris_TMC_1"
MLP_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris/Iris_1_MLPbV_pred.csv"
PRED_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris/Iris_1_Pred.csv"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Iris"
LABEL_COL = "target"
runtimes = {}
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
df_TMC = pd.read_csv(TMC_CSV)
tmc_vals = df_TMC["TMC_Shap"].to_numpy()
df_MLP = pd.read_csv(MLP_CSV)
mlp_vals = df_MLP["beta_pred"].to_numpy()
df_pred = pd.read_csv(PRED_CSV)
pred_vals = df_pred["Pred_Shap"].to_numpy()
per_vals = df_pred["PerDShap"].to_numpy()
drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()
X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()
train_ds = Dataset(X_train, y_train)
test_ds  = Dataset(X_test,  y_test)
N = len(train_ds)

# def timed_fit(name, val_obj, train_ds):
#     t0 = time.perf_counter()
#     val_obj.fit(train_ds)
#     dt = time.perf_counter() - t0
#     runtimes[name] = dt
#     print(f"[runtime] {name}: {dt:.3f} s")
#     return val_obj
#
# # ---------- Base models ----------
# # (A) Logistic regression for model-based methods
# lr = LogisticRegression(max_iter=200)
# scorer = SupervisedScorer("accuracy", test_ds, default=0.0, range=(0.0, 1.0))
# utility = ModelUtility(lr, scorer)
#
# # ---------- Common stoppers ----------
# TMC_MAX_ITER = 3 * N
# MC_BUDGET = 3 * N
# BURN_IN = max(200, int(0.4 * MC_BUDGET))
# RANK_RTOL = 1e-3
# shapley_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
# delta_stopping   = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
# group_stopping   = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
# banzhaf_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
# semi_stopping    = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
#
# # Parameter settings in Beta Shapley
# BETA_ALPHA, BETA_BETA = 1, 16
# beta_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
#
# # Parameter settings in Owen Shapley
# OWEN_TARGET_M = max(2000, int(38.4 * N))
# N_SAMPLES_INNER = 2
# N_SAMPLES_OUTER = int(np.ceil(OWEN_TARGET_M / N_SAMPLES_INNER))
#
# # Parameter settings in KNN Shapley
# KNN_K = min(10, max(1, N - 1))
#
# '''
# 2. Compute Valuations
# '''
# ''' A. Approximation of Data Shapley '''
# ' 1. TMC Shapley'
# model = 'logistic'
# problem = 'classification'
# metric = 'accuracy'
# directory = './temp'
# shapley_model = DShap(X_train, y_train,
#                       X_test, y_test,
#                       num_test = len(X_test), sources=None,
#                       sample_weight=None, model_family=model,
#                       metric=metric, overwrite=True,
#                       directory=directory, seed=251013,
#                       max_iter=TMC_MAX_ITER, tol=0.05)
# t0 = time.perf_counter()
# shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
# runtimes["TMC Shapley"] = time.perf_counter() - t0
# print(f"[runtime] TMC Shapley: {runtimes['TMC Shapley']:.3f} s")
# tmc_vals = np.asarray(shapley_model.vals_tmc).reshape(-1)
# Vehicle_TMC = pd.DataFrame(X_train)  # columns become 0..d-1
# Vehicle_TMC["TMC_Shap"] = tmc_vals
#
#
# '2. Delta Shapley'
# l, u = N // 3, (2 * N) // 3
# size_strategy = PowerLawSampleSize(lower_bound=l, upper_bound=u, exponent=0.0, n_samples = MC_BUDGET)
# sampler = StratifiedSampler(sample_sizes=size_strategy, seed=42)
# val_delta = DeltaShapleyValuation(utility=utility, sampler=sampler, is_done=delta_stopping, progress=True)
# timed_fit("DeltaShapley", val_delta, train_ds)
# delta_vals = val_delta.result.values
#
#
# '3. Group-Testing Shapley'
# val_group = GroupTestingShapleyValuation(utility=utility, progress=True, n_samples = MC_BUDGET, epsilon=0.05)
# timed_fit("GroupTesting", val_group, train_ds)
# group_vals = val_group.result.values
#
#
# '4. Owen Shapley'
# owen_sampler = OwenSampler(outer_sampling_strategy=GridOwenStrategy(n_samples_outer=N_SAMPLES_OUTER), n_samples_inner=N_SAMPLES_INNER)
# owen_stop = NoStopping(owen_sampler)
# owen_val = ShapleyValuation(utility, owen_sampler, owen_stop, progress=True)
#
# from joblib import parallel_config
# t0 = time.perf_counter()
# with parallel_config(n_jobs=4):
#     owen_val.fit(train_ds)
# runtimes["OwenShapley"] = time.perf_counter() - t0
# print(f"[runtime] OwenShapley: {runtimes['OwenShapley']:.3f} s")
# res_owen = owen_val.result.values
#
#
#
# ''' B. Generalisation of Data Shapley '''
# '5. Beta Shapley'
# # Use recommended settings from original paper: alpha=1, beta=16
# val_beta = BetaShapleyValuation(
#     utility=utility,
#     sampler=PermutationSampler(),
#     is_done=beta_stopping,
#     alpha=BETA_ALPHA,
#     beta=BETA_BETA,
#     progress=True
# )
# timed_fit("BetaShapley", val_beta, train_ds)
# beta_vals = val_beta.result.values
#
#
# '6. Data Banzhaf'
# val_banzhaf = BanzhafValuation(
#     utility=utility,
#     sampler=PermutationSampler(),
#     is_done=banzhaf_stopping,
#     progress=True
# )
# timed_fit("DataBanzhaf", val_banzhaf, train_ds)
# banzhaf_vals = val_banzhaf.result.values
#
#
# '7. Semi-Values'
# class UniformSemiValue(SemivalueValuation):
#     def log_coefficient(self, subset_size: int, total_size: int) -> float:
#         return 0.0  # log(1.0) for uniform weight
# val_semiv = UniformSemiValue(
#     utility=utility,
#     sampler=PermutationSampler(),
#     is_done=semi_stopping,
#     progress=True
# )
# timed_fit("UniformSemiValue", val_semiv, train_ds)
# semivalue_vals = val_semiv.result.values
#
#
#
# ''' C. Special Case of Data Shapley '''
# '8. KNN Shapley'
# from pydvl.valuation.methods.knn_shapley import KNNShapleyValuation
# # Practical default based on baseline literature (K=10)
# knn_model = KNeighborsClassifier(n_neighbors=KNN_K)
# val_knn = KNNShapleyValuation(model=knn_model, test_data=test_ds, progress=True)
# from joblib import parallel_config
# t0 = time.perf_counter()
# with parallel_config(n_jobs=4):
#     val_knn.fit(train_ds)
# runtimes["KNNShapley"] = time.perf_counter() - t0
# print(f"[runtime] KNNShapley: {runtimes['KNNShapley']:.3f} s")
# knn_vals = val_knn.result.values
#
# '9. Leave-one-out'
# clf_full = clone(lr)
# clf_full.fit(X_train, y_train)
# full_score = accuracy_score(y_test, clf_full.predict(X_test))
# loo_vals = np.empty(len(X_train), dtype=float)
# for i in range(len(X_train)):
#     mask = np.ones(len(X_train), dtype=bool)
#     mask[i] = False
#     clf_sub = clone(lr)
#     clf_sub.fit(X_train[mask], y_train[mask])
#     score_i = accuracy_score(y_test, clf_sub.predict(X_test))
#     loo_vals[i] = full_score - score_i
#
#
#
# ''' D. Influence Functions '''
# '10. CG Influence'
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from pydvl.influence.torch import CgInfluence
# class TorchLogistic(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.lin = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.lin(x)
#
# # Prepare Tensors & DataLoader
# X_train_t = torch.from_numpy(X_train).float()
# y_train_t = torch.from_numpy(y_train).long()
# train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
# X_test_t = torch.from_numpy(X_test).float()
# y_test_t = torch.from_numpy(y_test).long()
#
# # Instantiate model and loss
# input_dim = X_train.shape[1]
# num_classes = len(np.unique(y_train))
# torch_model = TorchLogistic(input_dim, num_classes)
# loss_fn = torch.nn.CrossEntropyLoss()
#
# optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
# for _ in range(100):
#     for bx, by in train_loader:
#         optimizer.zero_grad()
#         out = torch_model(bx)
#         loss = loss_fn(out, by)
#         loss.backward()
#         optimizer.step()
# infl_cg = CgInfluence(
#     model=torch_model,
#     loss=loss_fn,
#     regularization=1e-3,
#     rtol=1e-5,
#     atol=1e-6,
#     maxiter=None,
#     progress=True
# )
# # Fit influence model
# infl_cg.fit(train_loader)
# # Compute influence of training points on test set
# # result shape: (n_train, n_test)
# infl_matrix = infl_cg.influences(X_test_t, y_test_t, X_train_t, y_train_t)
# cg_influence_vals = infl_matrix.sum(axis=1).detach().numpy()
#
#
# ''' Collect and Save'''
# shapley_df = pd.DataFrame({
#     "TMC Shapley": tmc_vals,
#     "Delta Shapley": delta_vals,
#     "Group-Testing Shapley": group_vals,
#     "Owen Shapley": res_owen,
#     "Beta Shapley": beta_vals,
#     "Data Banzhaf": banzhaf_vals,
#     "Uniform SemiValue": semivalue_vals,
#     "KNN Shapley": knn_vals,
#     "LOO": loo_vals,
#     "CG Influence": cg_influence_vals
# })
#
#
# # Save runtime
# pd.DataFrame([runtimes]).to_csv(os.path.join(OUT_DIR, "volkert_1245_1_runtimes.csv"), index=False)
#
# # Save combined predictions and valuations
# df_train = df_train.reset_index(drop=True)
# combined_df = pd.concat([df_train, shapley_df], axis=1)
# combined_df.to_csv(os.path.join(OUT_DIR, "volkert_1245_1_all_baselines.csv"), index=False)


# ============================================================
#  X. Evaluation & Plots (MSE/RMSE, Paired plots, Removal/Addition)
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# -------------------------
# 0) (Optional) CSV containing other baselines besides TMC/Pred/MLP/PerDShap
#    If you already saved all those baselines into one csv, set it here.
#    Must contain: Delta/Group/Owen/Beta/Banzhaf/Semi/KNN  (and optionally others).
# -------------------------
BASELINES_ALL_CSV = os.path.join(OUT_DIR, "Iris_1_all_baselines.csv")  # <-- CHANGE if needed
if not os.path.exists(BASELINES_ALL_CSV):
    raise FileNotFoundError(
        f"Cannot find BASELINES_ALL_CSV:\n{BASELINES_ALL_CSV}\n"
        "Set BASELINES_ALL_CSV to your saved baselines csv which contains columns for Delta/Group/Owen/Beta/Banzhaf/Semi/KNN."
    )
df_base = pd.read_csv(BASELINES_ALL_CSV)

# -------------------------
# 1) small utilities
# -------------------------
def _to_1d_numeric(a):
    x = np.asarray(a).reshape(-1)
    # pd.to_numeric returns ndarray for ndarray input, Series for Series input
    return np.asarray(pd.to_numeric(x, errors="coerce"))

def _check_len(name, v, n):
    if len(v) != n:
        raise ValueError(f"Length mismatch: {name} has {len(v)} but expected {n}")

def compute_mse_rmse_vs_truth(truth, methods_dict):
    truth = _to_1d_numeric(truth)
    rows = []
    for name, vals in methods_dict.items():
        vals = _to_1d_numeric(vals)
        m = (~np.isnan(truth)) & (~np.isnan(vals))
        if m.sum() == 0:
            mse = np.nan
            rmse = np.nan
        else:
            diff = truth[m] - vals[m]
            mse = float(np.mean(diff ** 2))
            rmse = float(np.sqrt(mse))
        rows.append({"method": name, "mse": mse, "rmse": rmse, "n_valid": int(m.sum())})
    return pd.DataFrame(rows).sort_values("mse", ascending=True).reset_index(drop=True)

def _bounds_for_identity(x, ys):
    vv = [np.asarray(pd.to_numeric(x, errors="coerce"))]
    for y in ys:
        vv.append(np.asarray(pd.to_numeric(y, errors="coerce")))
    vv = np.concatenate(vv)
    vv = vv[~np.isnan(vv)]
    if vv.size == 0:
        return None
    lo, hi = float(np.min(vv)), float(np.max(vv))
    pad = 0.01 * (hi - lo + 1e-12)
    return lo, hi, pad

def plot_all_in_one_vs_truth(truth, series_dict, *, y_class=None, out_path="all_in_one.png", title="All methods vs PerDShap"):
    """
    One scatter plot:
      x = PerDShap (truth)
      y = each method value
    Identity line included.
    If y_class provided, marker encodes class.
    """
    x = _to_1d_numeric(truth)
    valid_x = ~np.isnan(x)

    ys_list = [_to_1d_numeric(v) for v in series_dict.values()]
    bounds = _bounds_for_identity(x[valid_x], [y[valid_x] for y in ys_list])
    if bounds is None:
        print("[WARN] skip all-in-one plot: no valid points")
        return
    lo, hi, pad = bounds
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(9.2, 9.2), dpi=250)

    # basic colors (do not over-engineer)
    names = list(series_dict.keys())
    # use matplotlib default cycle by not setting colors explicitly

    if y_class is None:
        for name, y in series_dict.items():
            yv = _to_1d_numeric(y)
            m = valid_x & ~np.isnan(yv)
            ax.scatter(x[m], yv[m], s=15, alpha=0.6, label=name)
    else:
        y_class = np.asarray(y_class).reshape(-1)
        classes = np.unique(y_class[valid_x])
        markers = ['o','s','^','D','P','X','v','<','>','*']
        # plot each method as a separate color (auto) but class marker for points
        for mi, (name, y) in enumerate(series_dict.items()):
            yv = _to_1d_numeric(y)
            for ci, c in enumerate(classes):
                mk = markers[ci % len(markers)]
                m = valid_x & (y_class == c) & ~np.isnan(yv)
                if np.any(m):
                    ax.scatter(x[m], yv[m], s=16, alpha=0.55, marker=mk, label=None)
        # legend only for method names (proxy handles)
        handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=7, label=name) for name in names]
        ax.legend(handles=handles, loc="upper left", fontsize=9, title="Series")

    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("PerDShap (ground truth)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if y_class is None:
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_paired_vs_truth(truth, pred, baseline, *,
                         y_class=None,
                         pred_name="Predicted",
                         base_name="Baseline",
                         out_path="paired.png",
                         title=None):
    """
    One plot overlays:
      - pred vs truth
      - baseline vs truth
    """
    x = _to_1d_numeric(truth)
    y_pred = _to_1d_numeric(pred)
    y_base = _to_1d_numeric(baseline)

    m = (~np.isnan(x)) & (~np.isnan(y_pred)) & (~np.isnan(y_base))
    if not np.any(m):
        print(f"[WARN] skip paired plot {base_name}: no valid points")
        return

    bounds = _bounds_for_identity(x[m], [y_pred[m], y_base[m]])
    if bounds is None:
        print(f"[WARN] skip paired plot {base_name}: cannot bound")
        return
    lo, hi, pad = bounds
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=250)

    if y_class is None:
        ax.scatter(x[m], y_pred[m], s=18, alpha=0.8, label=f"{pred_name} vs PerDShap")
        ax.scatter(x[m], y_base[m], s=18, alpha=0.8, label=f"{base_name} vs PerDShap")
    else:
        y_class = np.asarray(y_class).reshape(-1)
        classes = np.unique(y_class[m])
        markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']

        # Keep one color for Pred and one color for Baseline
        pred_color = "C0"
        base_color = "C1"

        for i, c in enumerate(classes):
            mc = m & (y_class == c)
            mk = markers[i % len(markers)]

            ax.scatter(x[mc], y_pred[mc], s=22, alpha=0.7, marker=mk,
                       color=pred_color, label=None)
            ax.scatter(x[mc], y_base[mc], s=22, alpha=0.7, marker=mk,
                       color=base_color, label=None)

        # Legend proxies (only 2 entries)
        ax.scatter([], [], s=40, marker='o', alpha=0.8, color=pred_color,
                   label=f"{pred_name} vs PerDShap")
        ax.scatter([], [], s=40, marker='o', alpha=0.8, color=base_color,
                   label=f"{base_name} vs PerDShap")

    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("PerDShap (ground truth)")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title if title is not None else f"{base_name} + {pred_name} vs PerDShap")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def model_builder():
    return LogisticRegression(max_iter=5000, multi_class="auto", n_jobs=-1, random_state=0)

def _utility_on_subset(X_sub, y_sub, X_test, y_test):
    if len(X_sub) == 0:
        return 0.0
    if len(np.unique(y_sub)) < 2:
        maj = np.bincount(y_test).argmax()
        return float(accuracy_score(y_test, np.full_like(y_test, maj)))
    clf = model_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_sub, y_sub)
    return float(accuracy_score(y_test, clf.predict(X_test)))

def curve_removal(values, X_train, y_train, X_test, y_test, *, order, step=0.05, max_fraction=0.4):
    vals = _to_1d_numeric(values)
    n = len(vals)
    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]
    elif order != "ascending":
        raise ValueError("order must be 'ascending' or 'descending'")

    fracs = np.arange(0.0, max_fraction + 1e-12, step)
    all_idx = np.arange(n)
    util = []
    for f in fracs:
        k = int(round(f * n))
        k = min(max(k, 0), n)
        rm = order_idx[:k]
        mask = np.ones(n, dtype=bool)
        mask[rm] = False
        keep = all_idx[mask]
        util.append(_utility_on_subset(X_train[keep], y_train[keep], X_test, y_test))
    return fracs, np.asarray(util, dtype=float)

def curve_addition(values, X_train, y_train, X_test, y_test, *, order, step=0.05, max_fraction=1.0):
    vals = _to_1d_numeric(values)
    n = len(vals)
    order_idx = np.argsort(vals)
    if order == "descending":
        order_idx = order_idx[::-1]
    elif order != "ascending":
        raise ValueError("order must be 'ascending' or 'descending'")

    fracs = np.arange(0.0, max_fraction + 1e-12, step)
    util = []
    for f in fracs:
        k = int(round(f * n))
        k = min(max(k, 1), n)
        sub = order_idx[:k]
        util.append(_utility_on_subset(X_train[sub], y_train[sub], X_test, y_test))
    return fracs, np.asarray(util, dtype=float)

def curve_random_removal(X_train, y_train, X_test, y_test, *, step=0.05, max_fraction=0.4, iters=10, seed=0):
    n = len(X_train)
    fracs = np.arange(0.0, max_fraction + 1e-12, step)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    U = np.zeros((iters, len(fracs)), dtype=float)
    for t in range(iters):
        perm = rng.permutation(n)
        for j, f in enumerate(fracs):
            k = int(round(f * n))
            k = min(max(k, 0), n)
            rm = perm[:k]
            mask = np.ones(n, dtype=bool)
            mask[rm] = False
            keep = all_idx[mask]
            U[t, j] = _utility_on_subset(X_train[keep], y_train[keep], X_test, y_test)
    return fracs, U.mean(axis=0)

def curve_random_addition(X_train, y_train, X_test, y_test, *, step=0.05, max_fraction=1.0, iters=10, seed=0):
    n = len(X_train)
    fracs = np.arange(0.0, max_fraction + 1e-12, step)
    rng = np.random.default_rng(seed)

    U = np.zeros((iters, len(fracs)), dtype=float)
    for t in range(iters):
        perm = rng.permutation(n)
        for j, f in enumerate(fracs):
            k = int(round(f * n))
            k = min(max(k, 1), n)
            sub = perm[:k]
            U[t, j] = _utility_on_subset(X_train[sub], y_train[sub], X_test, y_test)
    return fracs, U.mean(axis=0)

def plot_three_lines(x_fracs, rand_u, pred_u, base_u, *,
                     xlabel, ylabel, title, out_path,
                     rand_label="Random", pred_label="Predicted", base_label="Baseline"):
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=250)
    x = x_fracs * 100.0
    ax.plot(x, rand_u * 100.0, linestyle="--", marker="o", linewidth=2.0, label=rand_label)
    ax.plot(x, pred_u * 100.0, linestyle="-",  marker="o", linewidth=2.0, label=pred_label)
    ax.plot(x, base_u * 100.0, linestyle="-",  marker="o", linewidth=2.0, label=base_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# 2) Ground truth + methods arrays (from your already-loaded dfs)
# -------------------------
if "PerDShap" not in df_train.columns:
    raise ValueError("df_train must contain column 'PerDShap' (Permutation-based ground truth).")
per_vals = df_train["PerDShap"].to_numpy()
y_class = df_train[LABEL_COL].to_numpy() if LABEL_COL in df_train.columns else y_train

# pred_vals, mlp_vals, tmc_vals already loaded by you
# Ensure numeric
pred_vals = _to_1d_numeric(pred_vals)
mlp_vals  = _to_1d_numeric(mlp_vals)
tmc_vals  = _to_1d_numeric(tmc_vals)
per_vals  = _to_1d_numeric(per_vals)

n = len(X_train)
_check_len("PerDShap", per_vals, n)
_check_len("Pred", pred_vals, n)
_check_len("MLP", mlp_vals, n)
_check_len("TMC", tmc_vals, n)

# -------------------------
# 3) Load remaining baseline columns from df_base
# -------------------------
BASELINE_COLS = {
    "Delta Shapley":          "Delta Shapley",
    "Group-Testing Shapley":  "Group-Testing Shapley",
    "Owen Shapley":           "Owen Shapley",
    "Beta Shapley":           "Beta Shapley",
    "Data Banzhaf":           "Data Banzhaf",
    "Uniform SemiValue":      "Uniform SemiValue",
    "KNN Shapley":            "KNN Shapley",
    # NOTE: LOO/CG not requested now; you asked 1-8 + MLP
}

missing = [col for col in BASELINE_COLS.values() if col not in df_base.columns]
if len(missing) > 0:
    raise ValueError(
        f"Missing columns in {BASELINES_ALL_CSV}:\n{missing}\n\n"
        f"Available columns head:\n{list(df_base.columns)[:60]}"
    )

delta_vals = _to_1d_numeric(df_base[BASELINE_COLS["Delta Shapley"]].to_numpy())
group_vals = _to_1d_numeric(df_base[BASELINE_COLS["Group-Testing Shapley"]].to_numpy())
owen_vals  = _to_1d_numeric(df_base[BASELINE_COLS["Owen Shapley"]].to_numpy())
beta_vals  = _to_1d_numeric(df_base[BASELINE_COLS["Beta Shapley"]].to_numpy())
banzhaf_vals = _to_1d_numeric(df_base[BASELINE_COLS["Data Banzhaf"]].to_numpy())
semivalue_vals = _to_1d_numeric(df_base[BASELINE_COLS["Uniform SemiValue"]].to_numpy())
knn_vals = _to_1d_numeric(df_base[BASELINE_COLS["KNN Shapley"]].to_numpy())

for nm, v in [
    ("Delta Shapley", delta_vals),
    ("Group-Testing Shapley", group_vals),
    ("Owen Shapley", owen_vals),
    ("Beta Shapley", beta_vals),
    ("Data Banzhaf", banzhaf_vals),
    ("Uniform SemiValue", semivalue_vals),
    ("KNN Shapley", knn_vals),
]:
    _check_len(nm, v, n)

# -------------------------
# 4) Define method sets
#    Algorithms 1..8 + MLP:
#       1 TMC, 2 Delta, 3 Group, 4 Owen, 5 Beta, 6 Banzhaf, 7 Semi, 8 KNN, + MLP
# -------------------------
methods_1_to_8_plus_mlp = {
    "TMC Shapley":            tmc_vals,
    "Delta Shapley":          delta_vals,
    "Group-Testing Shapley":  group_vals,
    "Owen Shapley":           owen_vals,
    "Beta Shapley":           beta_vals,
    "Data Banzhaf":           banzhaf_vals,
    "Uniform SemiValue":      semivalue_vals,
    "KNN Shapley":            knn_vals,
    "MLP":                    mlp_vals,
}

# -------------------------
# 5) MSE/RMSE for (1..8 + MLP)
# -------------------------
methods_for_mse = dict(methods_1_to_8_plus_mlp)
methods_for_mse["Predicted"] = pred_vals
mse_rmse_df = compute_mse_rmse_vs_truth(per_vals, methods_for_mse)
print("\n[MSE/RMSE] (1..8 + MLP + Predicted) vs PerDShap")
print(mse_rmse_df)

os.makedirs(OUT_DIR, exist_ok=True)
mse_path = os.path.join(OUT_DIR, "mse_rmse_1to8_plus_MLP_vs_PerDShap.csv")
mse_rmse_df.to_csv(mse_path, index=False)
print("Saved:", mse_path)

# -------------------------
# 6) Linear comparison plots
#    6.1 ALL-IN-ONE: Pred + MLP + (1..8) vs PerDShap
# -------------------------
lin_dir = os.path.join(OUT_DIR, "LinearComparison_vs_PerDShap")
os.makedirs(lin_dir, exist_ok=True)

series_all = {
    "Predicted": pred_vals,
    "MLP":       mlp_vals,
    **{k: v for k, v in methods_1_to_8_plus_mlp.items() if k not in {"MLP"}},  # keep 1..8 too
}

all_in_one_path = os.path.join(lin_dir, "AllInOne_Pred_MLP_1to8_vs_PerDShap.png")
plot_all_in_one_vs_truth(
    per_vals, series_all,
    y_class=y_class,
    out_path=all_in_one_path,
    title="All methods (Pred + MLP + 1..8) vs PerDShap"
)
print("Saved:", all_in_one_path)

#    6.2 PAIRED plots: for each method in (1..8 + MLP),
#        overlay (method vs PerDShap) and (Pred vs PerDShap)
paired_dir = os.path.join(lin_dir, "PairedPlots")
os.makedirs(paired_dir, exist_ok=True)

paired_methods = {
    "TMC Shapley":            tmc_vals,
    "Delta Shapley":          delta_vals,
    "Group-Testing Shapley":  group_vals,
    "Owen Shapley":           owen_vals,
    "MLP Shapley": mlp_vals,
}

for name, base_vals in paired_methods.items():
    out_png = os.path.join(
        paired_dir,
        f"Paired_{name.replace(' ', '_').replace('-', '_')}_and_Pred_vs_PerDShap.png"
    )
    plot_paired_vs_truth(
        per_vals, pred_vals, base_vals,
        y_class=y_class,
        pred_name="Predicted",
        base_name=name,
        out_path=out_png,
        title=f"{name} + Predicted vs PerDShap"
    )
    print("Saved:", out_png)

# -------------------------
# 7) Removal & Addition plots for (1..8 + MLP)
#    Each plot has 3 lines: Random + Pred + ThisMethod
#    We produce 4 plots per method: removal(desc/asc) + addition(desc/asc)
# -------------------------
ra_dir = os.path.join(OUT_DIR, "RemovalAddition_3lines_1to8_plus_MLP")
os.makedirs(ra_dir, exist_ok=True)

STEP_REM = 0.05
MAX_REM  = 0.40
STEP_ADD = 0.05
MAX_ADD  = 1.00
RAND_ITERS = 10
RAND_SEED  = 0

# Random reference (shared)
f_rem, rand_rem = curve_random_removal(X_train, y_train, X_test, y_test,
                                       step=STEP_REM, max_fraction=MAX_REM, iters=RAND_ITERS, seed=RAND_SEED)
f_add, rand_add = curve_random_addition(X_train, y_train, X_test, y_test,
                                        step=STEP_ADD, max_fraction=MAX_ADD, iters=RAND_ITERS, seed=RAND_SEED)

# Pred curves (shared)
f_rem_p, pred_rem_desc = curve_removal(pred_vals, X_train, y_train, X_test, y_test,
                                       order="descending", step=STEP_REM, max_fraction=MAX_REM)
f_rem_p2, pred_rem_asc = curve_removal(pred_vals, X_train, y_train, X_test, y_test,
                                       order="ascending", step=STEP_REM, max_fraction=MAX_REM)
f_add_p, pred_add_desc = curve_addition(pred_vals, X_train, y_train, X_test, y_test,
                                        order="descending", step=STEP_ADD, max_fraction=MAX_ADD)
f_add_p2, pred_add_asc = curve_addition(pred_vals, X_train, y_train, X_test, y_test,
                                        order="ascending", step=STEP_ADD, max_fraction=MAX_ADD)
assert np.allclose(f_rem, f_rem_p) and np.allclose(f_rem, f_rem_p2)
assert np.allclose(f_add, f_add_p) and np.allclose(f_add, f_add_p2)

for name, base_vals in methods_1_to_8_plus_mlp.items():
    # removal curves for method
    f1, base_rem_desc = curve_removal(base_vals, X_train, y_train, X_test, y_test,
                                      order="descending", step=STEP_REM, max_fraction=MAX_REM)
    f2, base_rem_asc  = curve_removal(base_vals, X_train, y_train, X_test, y_test,
                                      order="ascending", step=STEP_REM, max_fraction=MAX_REM)
    assert np.allclose(f1, f_rem) and np.allclose(f2, f_rem)

    out1 = os.path.join(ra_dir, f"Removal_desc_{name.replace(' ', '_').replace('-', '_')}.png")
    plot_three_lines(
        f_rem, rand_rem, pred_rem_desc, base_rem_desc,
        xlabel="Percentage of training data removed (%)",
        ylabel="Accuracy (%)",
        title=f"Removal (desc): Random vs Predicted vs {name}",
        out_path=out1,
        pred_label="Predicted",
        base_label=name
    )
    print("Saved:", out1)

    out2 = os.path.join(ra_dir, f"Removal_asc_{name.replace(' ', '_').replace('-', '_')}.png")
    plot_three_lines(
        f_rem, rand_rem, pred_rem_asc, base_rem_asc,
        xlabel="Percentage of training data removed (%)",
        ylabel="Accuracy (%)",
        title=f"Removal (asc): Random vs Predicted vs {name}",
        out_path=out2,
        pred_label="Predicted",
        base_label=name
    )
    print("Saved:", out2)

    # addition curves for method
    f3, base_add_desc = curve_addition(base_vals, X_train, y_train, X_test, y_test,
                                       order="descending", step=STEP_ADD, max_fraction=MAX_ADD)
    f4, base_add_asc  = curve_addition(base_vals, X_train, y_train, X_test, y_test,
                                       order="ascending", step=STEP_ADD, max_fraction=MAX_ADD)
    assert np.allclose(f3, f_add) and np.allclose(f4, f_add)

    out3 = os.path.join(ra_dir, f"Addition_desc_{name.replace(' ', '_').replace('-', '_')}.png")
    plot_three_lines(
        f_add, rand_add, pred_add_desc, base_add_desc,
        xlabel="Percentage of training data used (%)",
        ylabel="Accuracy (%)",
        title=f"Addition (desc): Random vs Predicted vs {name}",
        out_path=out3,
        pred_label="Predicted",
        base_label=name
    )
    print("Saved:", out3)

    out4 = os.path.join(ra_dir, f"Addition_asc_{name.replace(' ', '_').replace('-', '_')}.png")
    plot_three_lines(
        f_add, rand_add, pred_add_asc, base_add_asc,
        xlabel="Percentage of training data used (%)",
        ylabel="Accuracy (%)",
        title=f"Addition (asc): Random vs Predicted vs {name}",
        out_path=out4,
        pred_label="Predicted",
        base_label=name
    )
    print("Saved:", out4)

print(f"\nDone. Created {2*len(methods_1_to_8_plus_mlp)} removal plots and {2*len(methods_1_to_8_plus_mlp)} addition plots.")

# -------------------------
# 8) Mislabelled detection plot (1 plot per dataset, all algorithms)
#    By default: synthetic noise indices (flip labels), detection uses "ascending" (low value = suspicious)
#    If you have real noisy mask, set is_noisy = ... (bool array of length n_train)
# -------------------------
def noisy_label_detection_curve(values, is_noisy, *, order="ascending"):
    vals = _to_1d_numeric(values)
    noisy = np.asarray(is_noisy).astype(bool)
    if len(vals) != len(noisy):
        raise ValueError("values and is_noisy must have same length.")
    n_noisy = int(noisy.sum())
    if n_noisy == 0:
        raise ValueError("No noisy labels found; cannot compute detection curve.")

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

def noisy_label_detection_plot_all_methods(vals_dict, *,
                                           is_noisy=None,
                                           order="ascending",
                                           synthetic_noise=True,
                                           noise_fraction=0.2,
                                           random_state=0,
                                           title="Mislabel detection",
                                           save_path="mislabel_detection.png"):
    if not isinstance(vals_dict, dict):
        raise ValueError("vals_dict must be a dict {name: values}")

    # infer n
    first = next(iter(vals_dict.values()))
    n = len(np.asarray(first).reshape(-1))

    # build noisy mask
    if is_noisy is None:
        if not synthetic_noise:
            raise ValueError("Provide is_noisy if synthetic_noise=False.")
        rng = np.random.default_rng(random_state)
        is_noisy = np.zeros(n, dtype=bool)
        k = max(1, int(round(noise_fraction * n)))
        idx = rng.choice(n, size=k, replace=False)
        is_noisy[idx] = True
    else:
        is_noisy = np.asarray(is_noisy).astype(bool)
        if len(is_noisy) != n:
            raise ValueError("is_noisy length mismatch")

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=250)

    y_true = is_noisy.astype(int)
    summary_rows = []

    # plot each method
    for name, v in vals_dict.items():
        v1 = _to_1d_numeric(v)
        if len(v1) != n:
            raise ValueError(f"Length mismatch in mislabel plot: {name} has {len(v1)} but expected {n}")

        frac, recall = noisy_label_detection_curve(v1, is_noisy, order=order)
        ax.plot(frac * 100, recall * 100, marker="o", linewidth=2.0, label=name)

        # AUC under recall-vs-fraction
        area = float(np.trapz(recall, frac))

        # ROC-AUC/AP by treating scores as noisy detector
        # if order="ascending": smaller values => more likely noisy => use scores = -v
        scores = (-v1) if order == "ascending" else v1
        try:
            roc = float(roc_auc_score(y_true, scores))
            ap = float(average_precision_score(y_true, scores))
        except ValueError:
            roc, ap = np.nan, np.nan

        summary_rows.append({"method": name, "AUC_recall": area, "ROC_AUC": roc, "AP": ap})

    # random baseline (dashed)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    noisy_perm = is_noisy[perm]
    cum = np.cumsum(noisy_perm.astype(int))
    frac_r = np.arange(1, n + 1) / n
    recall_r = cum / max(1, is_noisy.sum())
    ax.plot(frac_r * 100, recall_r * 100, linestyle="--", linewidth=2.0, label="Random")

    ax.set_xlabel("Percentage of training points inspected (%)")
    ax.set_ylabel("Noisy-label recall (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, ncol=1)
    ax.set_xlim(0, 50)  # inspect up to 50% by default (common in DV papers)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows).sort_values("AUC_recall", ascending=False).reset_index(drop=True)
    return summary

# Build methods for mislabel plot (all algorithms you care)
# Include PerDShap (truth) and Pred too; these are useful baselines in the detection plot.
mislabel_methods = {
    "PerDShap":   per_vals,
    "Predicted":  pred_vals,
    "MLP":        mlp_vals,
    **methods_1_to_8_plus_mlp,  # includes MLP too but ok if duplicated; keep unique keys
}
# remove duplicated key if any
mislabel_methods = {k: mislabel_methods[k] for k in dict.fromkeys(mislabel_methods.keys())}

mislabel_path = os.path.join(OUT_DIR, "MislabelDetection_AllMethods.png")
mislabel_summary = noisy_label_detection_plot_all_methods(
    mislabel_methods,
    is_noisy=None,                 # <-- replace with real noisy mask if you have
    synthetic_noise=True,
    noise_fraction=0.2,
    order="ascending",
    random_state=0,
    title="Mislabel detection (all methods)",
    save_path=mislabel_path
)
print("Saved:", mislabel_path)

mislabel_summary_path = os.path.join(OUT_DIR, "mislabel_detection_summary.csv")
mislabel_summary.to_csv(mislabel_summary_path, index=False)
print("Saved:", mislabel_summary_path)
print("\n[Mislabel summary]\n", mislabel_summary)