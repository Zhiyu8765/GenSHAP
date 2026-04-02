''' 0. Import Packages '''
' For Calculation of Ground-Truth Data Shapley Values '
import math
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
from tqdm.auto import tqdm as tqdm_auto
from sklearn.exceptions import ConvergenceWarning

' For Calculation of TMC Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#os.chdir('/tmp/pycharm_project_236/DataShapley')
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
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Optional, Callable, Iterable

#%matplotlib inline
MEM_DIR = '../'

import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore specific sklearn future warning about multi_class='auto'
warnings.filterwarnings(
    "ignore",
    message="'multi_class' was deprecated",
    category=FutureWarning,
    module="sklearn"
)


''' 1. Function to Calculate True Data Shapley Values'''
import math
from itertools import combinations
from typing import Optional, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def DataShapley(X, y, X_test, y_test,
    clf: Optional[LogisticRegression] = None,
    score_fn: Optional[Callable[[Iterable, Iterable], float]] = None,
    desc: str = "Computing Exact Data Shapley Value (Memoization)"
):
    # Ensure Pandas objects for iloc
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.reset_index(drop=True)
    y = pd.Series(y)    if not isinstance(y, pd.Series)    else y.reset_index(drop=True)
    N = len(X)

    # Default classifier: multinomial-capable LR
    if clf is None:
        clf = LogisticRegression(multi_class="auto", solver="lbfgs", max_iter=1000)

    # Scoring selection
    if score_fn is not None:
        use_clf_score = False
    elif hasattr(clf, "score"):
        use_clf_score = True
    else:
        use_clf_score = False
        score_fn = accuracy_score
    if not use_clf_score and score_fn is None:
        score_fn = accuracy_score

    BASELINE = 0.0  # V(∅) and any subset with <2 unique labels

    # Memoized utilities
    utility_cache = {}

    def compute_utility(S_fset: frozenset) -> float:
        """Return V(S) for subset S (frozenset of indices)."""
        if S_fset in utility_cache:
            return utility_cache[S_fset]

        if len(S_fset) == 0:
            u = BASELINE
        else:
            idx = list(S_fset)
            X_s = X.iloc[idx]
            y_s = y.iloc[idx]

            if y_s.nunique() < 2:   # degenerate subset
                u = BASELINE
            else:
                est = clone(clf)
                est.fit(X_s, y_s)
                if use_clf_score:
                    u = est.score(X_test, y_test)
                else:
                    y_pred = est.predict(X_test)
                    u = score_fn(y_test, y_pred)

        utility_cache[S_fset] = u
        return u

    # ---------- Exact Shapley main loop (correct indentation) ----------
    DShap = np.zeros(N, dtype=float)
    all_idx = set(range(N))

    for i in tqdm(range(N), desc=desc):
        marginal_sum = 0.0
        others = tuple(sorted(all_idx - {i}))

        # s ranges over subset sizes from 0 to N-1 (subsets of 'others')
        for s in range(0, N):  # N-1 is the largest nontrivial subset size
            weight = 1.0 / (N * math.comb(N - 1, s))
            for S in combinations(others, s):
                S_fset   = frozenset(S)
                S_i_fset = S_fset | {i}
                U_s  = compute_utility(S_fset)
                U_si = compute_utility(S_i_fset)
                marginal_sum += weight * (U_si - U_s)

        DShap[i] = marginal_sum

    return DShap


''' 2. Function to Generate Subsets '''
def make_multiclass_split(df, label_col, classes = None, n_per_class = 1000, test_size = 0.5, random_state = 0, scale = True):
    if classes is None:
        classes = sorted(df[label_col].unique().tolist())

    frames = []
    for c in classes:
        grp = df[df[label_col] == c]
        if n_per_class is not None:
            grp = grp.sample(n = n_per_class, random_state = random_state, replace = False)
        frames.append(grp)
    data = pd.concat(frames, axis = 0).reset_index(drop=True)

    X_all = data.drop(columns = [label_col])
    y_all = data[label_col]

    y_all, uniques = pd.factorize(y_all, sort = True)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_all, pd.Series(y_all, name = label_col), test_size = test_size,
                                                      stratify = y_all, random_state = random_state)

    if scale:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns = X_tr_raw.columns)
        X_te = pd.DataFrame(scaler.transform(X_te_raw), columns = X_te_raw.columns)
    else:
        X_tr = X_tr_raw.copy()
        X_te = X_te_raw.copy()

    return X_tr, y_tr, X_te, y_te


''' 3. Calculation of Data Shapley Values '''
'1) Iris'
save_dir   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrueDShap/Iris'
directory  = './temp'
classes        = [0, 1, 2]
m_per_class    = 7
seed0          = 251029
n_trials       = 5
os.makedirs(save_dir, exist_ok=True)

# Load once
iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
y_full = pd.Series(iris.target, name='target')
iris_df = pd.concat([X_full, y_full], axis=1)

for i in range(1, n_trials + 1):
    rng = np.random.RandomState(seed0 + 10000 * i)

    # 1) Use your splitter ONLY to filter by classes and provide a pool (no scaling here)
    X_tr0, y_tr0, X_te0, y_te0 = make_multiclass_split(
        df=iris_df, label_col='target', classes=classes,
        n_per_class=None,               # take ALL from these classes
        test_size=0.5, random_state=seed0 + i, scale=False
    )
    # Pool back together (within chosen classes)
    X_pool = pd.concat([X_tr0, X_te0], axis=0).reset_index(drop=True)
    y_pool = pd.concat([y_tr0, y_te0], axis=0).reset_index(drop=True)

    # 2) Inline: pick exactly m_per_class per class for TRAIN; the rest -> TEST
    # (clip m if any class has < m available)
    m_eff = min(m_per_class, min((y_pool == c).sum() for c in classes))
    if m_eff < m_per_class:
        print(f"[Warn] Reducing m from {m_per_class} to {m_eff} due to class counts.")
    m_use = m_eff

    train_idx = []
    for c in classes:
        idx_c = np.where(y_pool.values == c)[0]
        rng.shuffle(idx_c)
        train_idx.append(idx_c[:m_use])
    train_idx = np.concatenate(train_idx)

    all_idx  = np.arange(len(y_pool))
    test_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)

    X_train_raw = X_pool.iloc[train_idx].copy()
    y_train     = y_pool.iloc[train_idx].copy()
    X_test_raw  = X_pool.iloc[test_idx].copy()
    y_test      = y_pool.iloc[test_idx].copy()

    # 3) Scale AFTER selection (fit on train, transform test)
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw.values), columns=X_train_raw.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test_raw.values),      columns=X_test_raw.columns)

    # Save test set
    Iris_test = X_test.copy()
    Iris_test['target'] = y_test.values
    Iris_test.to_csv(f'{save_dir}/Iris_test_{i}', index=False)

    # 4) Exact True Data Shapley (memoized, baseline=0)
    DShap_true = DataShapley(
        X_train, y_train, X_test, y_test,
        clf=None, score_fn=None, desc=f"Exact Data Shapley (Iris, trial {i})")
    assert DShap_true is not None and len(DShap_true) == len(X_train), "DataShapley failed."

    Iris_true = X_train.copy()
    Iris_true['target']    = y_train.values
    Iris_true['TrueDShap'] = DShap_true
    Iris_true.to_csv(f'{save_dir}/Iris_{i}', index=False)

    # 5) TMC Shapley (sanitize for None before plotting)
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test), sources=None,
        sample_weight=None, model_family='logistic',
        metric='accuracy', overwrite=True,
        directory=directory, seed=seed0 + i,
        max_iter=100, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=False, loo_run=True)

    Iris_TMC = X_train.copy()
    Iris_TMC['target']   = y_train.values
    Iris_TMC['TMC_Shap'] = shapley_model.vals_tmc
    Iris_TMC.to_csv(f'{save_dir}/Iris_TMC_{i}', index=False)

    # 6) Removal plots (coerce TMC None/NaN to 0.0 inline)
    df_train = Iris_true
    df_tmc   = Iris_TMC

    X_train_np = df_train.drop(columns=["target", "TrueDShap"], errors="ignore").values
    y_train_np = df_train["target"].values
    true_shap  = pd.to_numeric(pd.Series(df_train["TrueDShap"].values), errors="coerce").fillna(0.0).to_numpy()
    tmc_shap   = pd.to_numeric(pd.Series(list(df_tmc["TMC_Shap"].values)), errors="coerce").fillna(0.0).to_numpy()

    X_test_np  = Iris_test.drop(columns=["target"], errors="ignore").values
    y_test_np  = Iris_test["target"].values

    try_vals = {
        "True Data Shapley": true_shap,
        "TMC Shapley":       tmc_shap,
    }

    plot_dir = save_dir
    os.makedirs(plot_dir, exist_ok=True)

    out_png_high = os.path.join(plot_dir, f"Removal_Iris_high_{i}.png")
    _rem_perc, _res = PerformancePlot_Removal(
        vals=try_vals,
        X_train=X_train_np, y_train=y_train_np,
        X_test=X_test_np,   y_test=y_test_np,
        a=0.03, b=0.4, ylim=(50, 105),
        order="descending", random_iters=10, save_path=out_png_high
    )

    out_png_low = os.path.join(plot_dir, f"Removal_Iris_low_{i}.png")
    _rem_perc, _res = PerformancePlot_Removal(
        vals=try_vals,
        X_train=X_train_np, y_train=y_train_np,
        X_test=X_test_np,   y_test=y_test_np,
        a=0.03, b=0.4, ylim=(50, 105),
        order="ascending", random_iters=10, save_path=out_png_low
    )


' 2) Breast Cancer '
n_trials = 5
m = 20
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrueDShap/BC'

directory = './temp'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(directory, exist_ok=True)

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target, name='target')
bc_df = pd.concat([X, y], axis=1)
bc_0 = bc_df[bc_df.target == 0]
bc_1 = bc_df[bc_df.target == 1]

for i in range(1, n_trials + 1):
    bc0 = bc_0.sample(n=m, random_state=i)
    bc1 = bc_1.sample(n=m, random_state=i)
    bc = pd.concat([bc0, bc1], axis=0).reset_index(drop=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        bc.drop(columns='target'), bc.target, test_size=0.5, stratify=bc.target, random_state=i)
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled = scaler1.transform(X_test_raw)
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_train_pca)
    X_test = scaler2.transform(X_test_pca)
    X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])

    out_tr = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    out_tr['target'] = y_train.values
    out_te = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
    out_te['target'] = y_test.values
    out_te.to_csv(f'{save_dir}/bc_test_{i}', index=False)

    phi = DataShapley(X_train, y_train, X_test, y_test, clf=None, score_fn=None, desc=f"Exact Data Shapley (Breast Cancer, trial {i})")
    BC_true = X_train.copy()
    BC_true['target'] = y_train.values
    BC_true['TrueDShap'] = phi
    BC_true.to_csv(f'{save_dir}/BC_{i}', index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values, m,
        sources=None, sample_weight=None, model_family='logistic',
        metric='accuracy', overwrite=True, directory=directory,
        seed=250625 + i, max_iter=100, tol=1e-5)
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    out_tr_tmc = out_tr.copy()
    out_tr_tmc['TMC_Shap'] = shapley_model.vals_tmc
    out_tr_tmc.to_csv(f'{save_dir}/bc_TMC_{i}', index=False)

    print(f"[Trial {i}] CSV files saved.")

save_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
             "Data Valuation/DataShapley_Graph/"
             "Datasets/TrueDShap/BC")
plot_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
            "Data Valuation/DataShapley_Graph/"
            "Datasets/TrueDShap/BC")
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, n_trials + 1):
    train_f = os.path.join(save_dir, f"bc_{i}")
    test_f  = os.path.join(save_dir, f"bc_test_{i}")
    tmc_f   = os.path.join(save_dir, f"bc_TMC_{i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)
    df_tmc   = pd.read_csv(tmc_f)

    X_train = df_train.drop(columns=["target", "TrueDShap"], errors="ignore").values
    y_train = df_train["target"].values
    true_shap = df_train["TrueDShap"].values
    tmc_shap = df_tmc["TMC_Shap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_bc_high_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(10, 105),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_bc_low_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(30, 105),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")

' 3) MNIST '
n_trials     = 5
seed0        = 10231
classes_pool = list(range(10))
k_min, k_max = 2, 5

save_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
            "Data Valuation/DataShapley_Graph/"
            "Datasets/TrueDShap/MNIST")
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

    m = rng.integers(20, 21)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        seed_per_class = 10000 * i + c
        idx_c = y_pool[y_pool == c].sample(n=need_c, random_state= seed_per_class).index
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
    X_test_pca = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f'PC{i + 1}' for i in range(X_test_pca.shape[1])])

    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/MNIST_{class_tag}_test_{i}", index=False)

    DShap_true = DataShapley(
        X_train, y_train, X_test, y_test,
        clf=None, score_fn=None, desc=f"Exact Data Shapley (MNIST, trial {i})")
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['TrueDShap'] = DShap_true
    df_tr.to_csv(f"{save_dir}/MNIST_{class_tag}_{i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['TrueDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_tag}_TMC_{i}", index=False)

    print(f"[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")

for i, (class_tag, m) in enumerate(zip(trial_tags, trial_ms), start=1):
    base = f"{save_dir}/MNIST_{class_tag}"
    df_tr  = pd.read_csv(f"{base}_{i}")
    df_te  = pd.read_csv(f"{base}_test_{i}")
    df_tmc = pd.read_csv(f"{base}_TMC_{i}")

    X_tr = df_tr.drop(columns=['target', 'TrueDShap']).values
    y_tr = df_tr['target'].values
    true_shap = df_tr['TrueDShap'].values
    tmc_shap = df_tmc['TMC_Shap'].values

    X_te = df_te.drop(columns=['target']).values
    y_te = df_te['target'].values

    PerformancePlot_Removal(
        vals={
            'True Data Shapley': true_shap,
            'TMC Shapley': tmc_shap
        },
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(20, 105), order='descending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_tag}_high_{i}.png"
    )

    PerformancePlot_Removal(
        vals={
            'True Data Shapley': true_shap,
            'TMC Shapley': tmc_shap
        },
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(20, 105), order='ascending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_tag}_low_{i}.png"
    )

    print(f"[Trial {i}] Plots saved for tag={class_tag}, m={m}.")
