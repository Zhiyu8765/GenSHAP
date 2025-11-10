'''
0. Import Packages
'''
import os
import warnings
import numpy as np
import pandas as pd
import inspect
from math import ceil, sqrt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_22.csv"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_test_22.csv"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251109"
LABEL_COL = "target"
PRED_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_22_Pred.csv"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_TMC_22.csv"
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


# ---------- Collect & save ----------
shapley_df = pd.DataFrame({
    "Permutated_Shapley":    res_shapley,
    "BetaShapley":            res_beta,
    "DeltaShapley":           delta_vals,
    "GroupTesting":           group_vals,
    "OwenShapley":            res_owen,
    "Classwise_Shapley":       cs_values,
    "KNNShapley":             knn_vals
})
pred_df = pd.read_csv(PRED_CSV)
pred_df = pred_df.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
combined_df = pd.concat([pred_df, shapley_df], axis=1)
combined_df.to_csv(os.path.join(OUT_DIR, "MNIST_369xx_20_Comparison.csv"), index=False)


'''
3. Linear Comparison Plot
'''
# df = pd.read_csv(os.path.join(OUT_DIR, "MNIST_459xx_22_Comparison.csv"))
df = pd.read_csv(TRAIN_CSV)
df_tmc = pd.read_csv(TMC_CSV)
y_vals = df["target"].to_numpy()
per_vals = df["PerDShap"].to_numpy()
# beta_vals  = df["BetaShapley"].to_numpy()
# delta_vals = df["DeltaShapley"].to_numpy()
tmc_vals  = df_tmc["TMC_Shap"].to_numpy()
# gt_vals = df["GroupTesting"].to_numpy()
# owen_vals  = df["OwenShapley"].to_numpy()
# cw_vals = df["Classwise_Shapley"].to_numpy()
# knn_vals  = df["KNNShapley"].to_numpy()
# pred_vals = df["Pred_Shap"].to_numpy()

series = {
    # "Predicted Shapley":  pred_vals,
    "TMC Shapley":        tmc_vals,
    # "Beta Shapley":       beta_vals,
    # "Delta Shapley":      delta_vals,
    # "Group Testing":      gt_vals,
    # "Owen Shapley":       owen_vals,
    # "Classwise Shapley":  cw_vals,
    # "KNN Shapley":        knn_vals,
}
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
styles  = {c: dict(marker=MARKERS[i % len(MARKERS)], s=22, linewidths=1.2)
           for i, c in enumerate(classes)}
all_y = np.concatenate([pd.to_numeric(v, errors="coerce") for v in series.values()])
x = pd.to_numeric(per_vals, errors="coerce")
valid = ~np.isnan(x)
lo = np.nanmin(np.concatenate([x[valid], all_y[~np.isnan(all_y)]]))
hi = np.nanmax(np.concatenate([x[valid], all_y[~np.isnan(all_y)]]))
pad = 0.001 * (hi - lo + 1e-12)
fig, ax = plt.subplots(figsize=(9.2, 9.2))

for c in classes:
    m_class = (y_vals == c) & valid
    st = styles[c].copy()
    for name, y in series.items():
        y_num = pd.to_numeric(y, errors="coerce")
        m = m_class & ~np.isnan(y_num)
        if not np.any(m):
            continue
        ax.scatter(x[m], y_num[m], marker=st['marker'], s=st['s'],
                   facecolors=COLORS[name], edgecolors='k', alpha=0.6)

ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--k', linewidth=1.4)
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(lo - pad, hi + pad)
ax.set_xlabel('Permuted Shapley Value (PerDShap)')
ax.set_ylabel('Other Shapley Variants')
# ax.set_title('Linear Comparison — All Variants')
ax.grid(True, linestyle='--', alpha=0.5)

series_handles = [Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none',
                  markeredgecolor=COLORS[name], markersize=8, label=name) for name in series.keys()]
legend_series = ax.legend(handles=series_handles, title="Series (color)",loc='upper left', fontsize=9)
ax.add_artist(legend_series)

class_handles = [Line2D([0],[0], marker=styles[c]['marker'], linestyle='None',
                        markerfacecolor='none', markeredgecolor='k',
                        markersize=8, label=f'class {int(c)}')
                for c in classes]
legend_classes = ax.legend(handles=class_handles, title="Class (marker)", loc='lower right', fontsize=9)
ax.add_artist(legend_classes)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "MNIST_459xx_LinearCompare.png")
plt.savefig(out_png, dpi=500)
plt.show()
plt.close()
print("Saved linear comparison to:", out_png)

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

from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal, PerformancePlot_Addition
PERF_PATH = os.path.join(OUT_DIR, "MNIST_459xx_RemovalLow.png")
removal_percentages, results = PerformancePlot_Removal(
    vals = {'Permutated Shapley': per_vals},
    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.35, ylim = (90, 93), order = 'ascending',
    random_iters = 42, save_path = PERF_PATH
)
PERF_PATH2 = os.path.join(OUT_DIR, "MNIST_459xx_AddLow.png")
add_pct, add_results = PerformancePlot_Addition(
    vals = {"Permuted Data Shapley": per_vals},
    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, a = 0.05,
    b = 1.0, order = 'ascending', random_iters = 10, ylim = None, save_path = PERF_PATH2
)
