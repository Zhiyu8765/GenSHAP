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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
import warnings
from sklearn.exceptions import ConvergenceWarning
COMMON_FIGSIZE = (12, 12)
COMMON_DPI     = 500
plt.rcParams.update({
    "font.size": 10,           # general font size
    "axes.labelsize": 10,      # x/y label size
    "axes.titlesize": 10,    # title size
    "xtick.labelsize": 10,     # x tick labels
    "ytick.labelsize": 10,     # y tick labels
    "legend.fontsize": 10,     # legend text
    "figure.titlesize": 10     # figure title
})

'''
1. Linear Comparison Plot
'''
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Iris_Comparison/Iris_1"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Iris_Comparison/Iris_test_1"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251117/Review_Graphs"
LABEL_COL = "target"
# PRED_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I//Iris_Comparison/Iris_1"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Iris_Comparison/Iris_TMC_1"
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()
X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()
df = pd.read_csv(TRAIN_CSV)
df_tmc = pd.read_csv(TMC_CSV)
y_vals = df["target"].to_numpy()
per_vals = df["PerDShap"].to_numpy()
tmc_vals  = df_tmc["TMC_Shap"].to_numpy()

series = {"TMC Shapley": tmc_vals}
COLORS = {"TMC Shapley": "#ff7f0e"}
classes = np.unique(y_vals)
MARKERS = ['o','s','^','D','P','X','v','<','>','*']
styles  = {c: dict(marker=MARKERS[i % len(MARKERS)], s=100, linewidths=1.2)
           for i, c in enumerate(classes)}
all_y = np.concatenate([pd.to_numeric(v, errors="coerce") for v in series.values()])
x = pd.to_numeric(per_vals, errors="coerce")
valid = ~np.isnan(x)
lo = np.nanmin(np.concatenate([x[valid], all_y[~np.isnan(all_y)]]))
hi = np.nanmax(np.concatenate([x[valid], all_y[~np.isnan(all_y)]]))
pad = 0.1 * (hi - lo + 1e-12)
fig = plt.figure(figsize=COMMON_FIGSIZE, dpi=COMMON_DPI)
ax  = fig.add_axes([0.15, 0.15, 0.76, 0.76])
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
                  markeredgecolor=COLORS[name], markersize=15, label=name) for name in series.keys()]
# legend_series = ax.legend(handles=series_handles, title="Series (color)",loc='upper left', fontsize=15)
# ax.add_artist(legend_series)

class_handles = [Line2D([0],[0], marker=styles[c]['marker'], linestyle='None',
                        markerfacecolor='none', markeredgecolor='k',
                        markersize=15, label=f'class {int(c)}')
                for c in classes]
legend_classes = ax.legend(handles=class_handles, title="Class (marker)", loc='upper left', fontsize=25)
ax.add_artist(legend_classes)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "Iris_LinearCompare.pdf")
plt.savefig(out_png, dpi=500)
plt.show()
plt.close()
print("Saved linear comparison to:", out_png)


'''
2. Removal and Addition Plots
'''
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_22.csv"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_test_22.csv"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251117/Removal_Addition"
LABEL_COL = "target"
# PRED_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I//Iris_Comparison/Iris_1"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/MNIST_Comparison/MNIST_459xx_TMC_22.csv"

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()
X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()

df = pd.read_csv(TRAIN_CSV)
df_tmc = pd.read_csv(TMC_CSV)
y_vals = df["target"].to_numpy()
per_vals = df["PerDShap"].to_numpy()

from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal, PerformancePlot_Addition
PERF_PATH = os.path.join(OUT_DIR, "MNIST_459_RemovalHigh.pdf")
removal_percentages, results = PerformancePlot_Removal(
    vals = {'Permutated Shapley': per_vals},
    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.35, ylim = (85, 93), order = 'descending',
    random_iters = 42, save_path = PERF_PATH
)
PERF_PATH2 = os.path.join(OUT_DIR, "MNIST_459_AddLow.pdf")
add_pct, add_results = PerformancePlot_Addition(
    vals = {"Permuted Data Shapley": per_vals},
    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, a = 0.05,
    b = 1.0, order = 'ascending', random_iters = 10, ylim = None, save_path = PERF_PATH2
)


'''
3. Mislabelled Detection
'''
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251117/Review_Graphs"
from sklearn.exceptions import ConvergenceWarning
def Per_DShap(X, y, X_test, y_test, n_permutations = 3000,
              model_builder = lambda: LogisticRegression(max_iter = 200, multi_class = 'multinomial', solver='lbfgs', random_state=0),
              random_state = None, progress = True):
    X_np = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
    y_np = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    Xv = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else np.asarray(X_test)
    yv = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

    N = len(X_np)
    if len(y_np) != N:
        raise ValueError('X and y have different lengths!')

    baseline = 0.0  # when empty subset or single-class subset

    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = model_builder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    rng = np.random.default_rng(random_state)
    phi = np.zeros(N, dtype = float)
    counts = np.zeros(N, dtype = int)

    iterator = range(n_permutations)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc = "Permutation Data Shapley", unit = "perm")
        except Exception:
            pass

    for _ in iterator:
        perm = rng.permutation(N)
        prev_v = baseline
        for j in range(1, N+1):
            i = perm[j-1]
            S = perm[:j]
            vj = utility(S)
            mc = vj - prev_v
            counts[i] += 1
            phi[i] += (mc - phi[i]) / counts[i]
            prev_v = vj
    return phi
def make_noisy_labels(y, frac=0.20, rng=0):
    """Flip a fraction of training labels to a different class; return noisy_y and flipped indices."""
    rng = np.random.default_rng(rng)
    y = np.asarray(y).copy()
    n = len(y)
    k = int(np.round(frac * n))
    flip_idx = rng.choice(n, size=k, replace=False)
    classes = np.unique(y)
    for i in flip_idx:
        choices = classes[classes != y[i]]
        y[i] = rng.choice(choices)
    return y, flip_idx

def build_lr():
    return LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs', random_state=0)

def utility_acc(X_tr, y_tr, X_te, y_te):
    if len(np.unique(y_tr)) < 2:
        return (y_te == np.unique(y_tr)[0]).mean()
    clf = build_lr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
    return (clf.predict(X_te) == y_te).mean()

def LOO_values(X, y, X_test, y_test):
    """LOO[i] = V(D) - V(D\\{i})."""
    n = len(y)
    v_full = utility_acc(X, y, X_test, y_test)
    vals = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        v_minus = utility_acc(X[mask], y[mask], X_test, y_test)
        vals[i] = v_full - v_minus
    return vals

def detection_curve(order_idx, mislabeled_mask, max_frac=0.5, step_frac=0.02):
    """Inspect points from the beginning of order_idx; return (x%, y%) where y is fraction of mislabeled fixed."""
    n = len(order_idx)
    K = int(np.floor(max_frac * n))
    step = max(1, int(np.floor(step_frac * n)))
    counts = list(range(0, K + 1, step))
    total_mislabeled = mislabeled_mask.sum()
    y_curve = []
    for k in counts:
        if k == 0:
            y_curve.append(0.0)
            continue
        idx = order_idx[:k]
        found = mislabeled_mask[idx].sum()
        y_curve.append(0.0 if total_mislabeled == 0 else found / total_mislabeled)
    x = [100 * c / n for c in counts]
    y = [100 * v for v in y_curve]
    return np.array(x), np.array(y)
NOISE_FRAC = 0.20
RNG_SEED = 0
y_train_noisy, flipped_idx = make_noisy_labels(y_train, frac=NOISE_FRAC, rng=RNG_SEED)
mislabeled_mask = np.zeros_like(y_train_noisy, dtype=bool); mislabeled_mask[flipped_idx] = True
print(f"Injected mislabeled labels: {mislabeled_mask.sum()} / {len(y_train_noisy)} ({100*NOISE_FRAC:.1f}%)")

# Permutation-based Data Shapley
N_PERM = 3000   # adjust for time/variance trade-off
per_vals = Per_DShap(
    X=X_train, y=y_train_noisy,
    X_test=X_test, y_test=y_test,
    n_permutations=N_PERM, model_builder=build_lr,
    random_state=RNG_SEED, progress=True
)
# LOO
loo_vals = LOO_values(X_train, y_train_noisy, X_test, y_test)
# Orders
order_per = np.argsort(np.nan_to_num(per_vals, nan=np.inf))
order_loo = np.argsort(np.nan_to_num(loo_vals, nan=np.inf))
# Random
RANDOM_ITERS = 50
rng = np.random.default_rng(RNG_SEED)
random_curves = []
for _ in range(RANDOM_ITERS):
    order_rand = rng.permutation(len(y_train_noisy))
    _, y_r = detection_curve(order_rand, mislabeled_mask, max_frac=0.5, step_frac=0.02)
    random_curves.append(y_r)
# compute x and curves
x_per, y_per = detection_curve(order_per, mislabeled_mask, max_frac=0.5, step_frac=0.02)
x_loo, y_loo = detection_curve(order_loo, mislabeled_mask, max_frac=0.5, step_frac=0.02)
assert np.allclose(x_per, x_loo)
x = x_per
y_rand = np.mean(random_curves, axis=0)

# Plot
fig = plt.figure(figsize=COMMON_FIGSIZE, dpi=COMMON_DPI)
ax = fig.add_axes([0.15, 0.15, 0.76, 0.76])
# plt.title("Label-Noise Detection\nLogistic Regression", pad=14)
plt.plot(x, y_per,  marker='s',  linewidth=3, markersize=7, label="Permutated Shapley")
plt.plot(x, y_loo,  marker='^',  linewidth=3, markersize=7, label="LOO")
plt.plot(x, y_rand, linestyle=':', linewidth=3, label="Random")

plt.xlabel("Fraction of data checked (%)")
plt.ylabel("Fraction of incorrect labels fixed (%)")
plt.xlim(0, 50)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(
    loc='upper center',            # center horizontally
    bbox_to_anchor=(0.5, 1.20),    # position above the top axis
    ncol=3,                        # number of columns (1 row)
    frameon=False,                 # remove legend box background (optional)
    fontsize=25,                   # adjust font size
    handlelength=1.5,              # shorten legend line handles
    columnspacing=1.2              # spacing between entries
)

fig_path = os.path.join(OUT_DIR, "Iris_LabelNoise.PDF")
plt.tight_layout()
plt.savefig(fig_path, dpi=COMMON_DPI)
plt.show()
print(f"Saved figure: {fig_path}")


'''
4. Human Inspection
'''
n_trials     = 1
seed0        = 123
classes_pool = list(range(10))
k_min, k_max = 2, 3
save_dir = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/251117/Review_Graphs"
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

import tensorflow as tf
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
X_pool = X_train_full.reshape(len(X_train_full), -1)
y_pool = y_train_full
X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(y_pool, name='target')

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))

trial_tags = []
trial_ms   = []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    m = rng.integers(100, 101)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(n=need_c, random_state=rs + 1007 + c).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca = PCA(n_components=20, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f'PC{i + 1}' for i in range(X_test_pca.shape[1])])

    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/MNIST_{class_tag}_test_{i}", index=False)

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True)
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/MNIST_{class_tag}_{i}", index=False)

# build inverse map (add this once; label_map already exists above)
inv_label_map = {v: k for k, v in label_map.items()}

# original labels for the current train split (NO recomputation)
y_train_orig_np = y_train.map(inv_label_map).to_numpy()

# --- identify lowest and highest 3 ---
lowest_idx = np.argsort(phi)[:3]
highest_idx = np.argsort(phi)[-3:][::-1]

print("Lowest-value indices:", lowest_idx)
print("Highest-value indices:", highest_idx)
print("Lowest Shapley values:", phi[lowest_idx])
print("Highest Shapley values:", phi[highest_idx])

# --- reconstruct raw 28×28 images for visualization ---
X_train_raw_imgs = X_train_raw.to_numpy().reshape(-1, 28, 28)
y_train_np = y_train.to_numpy()

ordered_indices = list(lowest_idx) + list(highest_idx)
# ordered_indices = list(lowest_idx)
fig, axes = plt.subplots(1, 6, figsize=(3.4, 1.2), dpi=400)

plt.subplots_adjust(
    left=0.00, right=1.00,
    bottom=0.00, top=1.00,
    wspace=0.05, hspace=0.00
)

axes = np.atleast_1d(axes).ravel()

for j, ax in enumerate(axes):
    idx = ordered_indices[j]
    tag = f"Low-{j+1}" if j < 3 else f"High-{j-2}"

    ax.imshow(X_train_raw_imgs[idx], cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # --- Draw titles BELOW the image (solves bottom margin issue) ---
    ax.set_title(
        f"{tag}\nϕ={phi[idx]:.3f}\nLabel={y_train_orig_np[idx]}",
        fontsize=6,
        loc="center",
        pad=-3              # Negative pad pulls title DOWN into the image
    )

out_vis = os.path.join(save_dir, f"MNIST_{class_tag}_HumanInspect_{i}.pdf")
plt.savefig(out_vis, dpi=400, bbox_inches="tight", pad_inches=0.01)
plt.show()
print(f"[Trial {i}] Human inspection figure saved to: {out_vis}")