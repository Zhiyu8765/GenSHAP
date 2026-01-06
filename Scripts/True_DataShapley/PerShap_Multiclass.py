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
# os.chdir('/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/DataShapley')
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

''' 1. Function to Calculate Permutated-Based Data Shapley Values'''
def Per_DShap(X, y, X_test, y_test, n_permutations = 5000,
              model_builder = lambda: LogisticRegression(max_iter = 200, multi_class = 'multinomial'),
              random_state = None, progress = True):
    X_np = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
    y_np = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    Xv = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else np.asarray(X_test)
    yv = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

    N = len(X_np)
    if len(y_np) != N:
        raise ValueError('X and y have different lengths!')

    baseline = 0.0

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
            iterator = tqdm(iterator, desc = "Permutated-Based Data Shapley Values (Multi-class)", unit = "perm")
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
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/Iris_01'
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
directory = './temp'

iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, name = 'target')
iris_df = pd.concat([X, y], axis = 1)

X_train, y_train, X_test, y_test = make_multiclass_split(iris_df, label_col = 'target', classes = [0, 1, 2], n_per_class = 50,
                                                         test_size = 0.5, random_state = 1, scale = True)
Iris_test = X_test.copy()
Iris_test['target'] = y_test.values
Iris_test.to_csv(f'{save_dir}/Iris_test_1', index=False)
Per_Shap = Per_DShap(X_train, y_train, X_test, y_test, n_permutations = 5000)
Iris_per = X_train.copy()
Iris_per['target'] = y_train.values
Iris_per['PerDShap'] = Per_Shap
Iris_per.to_csv(f'{save_dir}/Iris_1', index=False)

shapley_model = DShap(X_train.values, y_train.values,
                      X_test.values, y_test.values,
                      num_test = 75, sources=None,
                      sample_weight=None, model_family=model,
                      metric=metric, overwrite=True,
                      directory=directory, seed=251013,
                      max_iter=100, tol=1e-5)
shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
Iris_TMC = X_train.copy()
Iris_TMC['TMC_Shap'] = shapley_model.vals_tmc
Iris_TMC.to_csv(f'{save_dir}/Iris_TMC_1', index=False)

save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/1000pts_I/Iris_01"
)
plot_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/1000pts_I/Iris_01"
)
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, 2):
    train_f = os.path.join(save_dir, f"Iris_{i}")
    test_f  = os.path.join(save_dir, f"Iris_test_{i}")
    tmc_f   = os.path.join(save_dir, f"Iris_TMC_{i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)
    df_tmc   = pd.read_csv(tmc_f)

    X_train = df_train.drop(columns=["target", "PerDShap"], errors="ignore").values
    y_train = df_train["target"].values
    per_shap = df_train["PerDShap"].values
    tmc_shap = df_tmc["TMC_Shap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_Iris_high_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "Permutated-Based Data Shapley": per_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(20, 105),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_Iris_low_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "Permutated-Based Data Shapley": per_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(50, 105),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")

' 2) Breast Cancer '
n_trials = 5
m = 200
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/BC'

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

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
                    model_builder=lambda: LogisticRegression(max_iter=100),
                    random_state=0, progress=True)
    BC_per = X_train.copy()
    BC_per['target'] = y_train.values
    BC_per['PerDShap'] = phi
    BC_per.to_csv(f'{save_dir}/BC_{i}', index=False)

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
             "Datasets/1000pts_I/BC")
plot_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
            "Data Valuation/DataShapley_Graph/"
            "Datasets/1000pts_I/BC")
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, n_trials + 1):
    train_f = os.path.join(save_dir, f"bc_{i}")
    test_f  = os.path.join(save_dir, f"bc_test_{i}")
    tmc_f   = os.path.join(save_dir, f"bc_TMC_{i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)
    df_tmc   = pd.read_csv(tmc_f)

    X_train = df_train.drop(columns=["target", "PerDShap"], errors="ignore").values
    y_train = df_train["target"].values
    per_shap = df_train["PerDShap"].values
    tmc_shap = df_tmc["TMC_Shap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_bc_high_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": per_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(10, 105),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_bc_low_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": per_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(30, 105),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")

' 3) MNIST '
n_trials     = 4
seed0        = 10231
classes_pool = list(range(10))
k_min, k_max = 2, 3

save_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
            "Data Valuation/DataShapley_Graph/"
            "Datasets/1000pts_I/MNIST")
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

    m = rng.integers(1000, 1001)
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
    df_te.to_csv(f"{save_dir}/MNIST_{class_tag}_test_{10+i}", index=False)

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True)
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/MNIST_{class_tag}_{10+i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_tag}_TMC_{10+i}", index=False)

    print(f"[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")

for i, (class_tag, m) in enumerate(zip(trial_tags, trial_ms), start=1):
    base = f"{save_dir}/MNIST_{class_tag}"
    df_tr  = pd.read_csv(f"{base}_{3+i}")
    df_te  = pd.read_csv(f"{base}_test_{3+i}")
    df_tmc = pd.read_csv(f"{base}_TMC_{3+i}")

    X_tr = df_tr.drop(columns=['target', 'PerDShap']).values
    y_tr = df_tr['target'].values
    per_shap = df_tr['PerDShap'].values
    tmc_shap = df_tmc['TMC_Shap'].values

    X_te = df_te.drop(columns=['target']).values
    y_te = df_te['target'].values

    PerformancePlot_Removal(
        vals={
            'Permutated Data Shapley': per_shap,
            'TMC Shapley': tmc_shap
        },
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(80, 105), order='descending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_tag}_high_{i}.png"
    )

    PerformancePlot_Removal(
        vals={
            'Permutated Data Shapley': per_shap,
            'TMC Shapley': tmc_shap
        },
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(80, 105), order='ascending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_tag}_low_{i}.png"
    )

    print(f"[Trial {i}] Plots saved for tag={class_tag}, m={m}.")


' 4) Vehicle '
save_dir = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/Vehicle"
plot_dir = save_dir
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
directory = './temp'
RANDOM_STATE = 23
test_size = 0.5

vehicle = fetch_openml(name = "vehicle", version = 1, as_frame=True)
X_full = vehicle.data
y_full = pd.factorize(vehicle.target, sort=True)[0]
vehicle_df = X_full.copy()
vehicle_df['target'] = y_full
print("vehicle counts:", vehicle_df['target'].value_counts().to_dict()) # Nearly balanced 4 classes

X_all = vehicle_df.drop(columns = ['target'])
y_all = vehicle_df['target']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size,
                                                                    stratify=y_all, random_state=RANDOM_STATE)
cat_cols = X_train_raw.select_dtypes(include = ["object", "category"]).columns.tolist()
num_cols = X_train_raw.columns.difference(cat_cols).tolist()
one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
pre_cols = ColumnTransformer(transformers = [("num", StandardScaler(with_mean = True, with_std = True), num_cols),
                                             ("cat", one_hot, cat_cols)], remainder = "drop", verbose_feature_names_out = False)
pipe = Pipeline(steps = [("pre", pre_cols),
                         # ("pca", PCA(n_components = 20, random_state=RANDOM_STATE)),
                         ("std2", StandardScaler(with_mean = True, with_std = True))])
X_train_arr = pipe.fit_transform(X_train_raw)
n_feats = X_train_arr.shape[1]
colnames = [f"f{i+1}" for i in range(n_feats)]

X_train = pd.DataFrame(X_train_arr, columns=colnames)

X_test_arr = pipe.transform(X_test_raw)
X_test = pd.DataFrame(X_test_arr, columns=colnames)
vehicle_test = X_test.copy()
vehicle_test['target'] = y_test.values
vehicle_test.to_csv(f'{save_dir}/vehicle_test_1', index=False)

Per_Shap = Per_DShap(X_train, y_train, X_test, y_test, n_permutations = 3*len(X_full),
                     model_builder = lambda: LogisticRegression(max_iter=200, multi_class = "auto", random_state = RANDOM_STATE),
                     random_state = RANDOM_STATE, progress = True)
Vehicle_per = X_train.copy()
Vehicle_per['target'] = y_train.values
Vehicle_per['PerDShap'] = Per_Shap
Vehicle_per.to_csv(f'{save_dir}/Vehicle_1', index=False)

shapley_model = DShap(
        X_train.values, y_train.values, X_test.values,  y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None, model_family=model,
        metric=metric, overwrite=True, directory=directory, seed=251013,
        max_iter=100, tol=1e-5)
shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
Vehicle_tmc = X_train.copy()
Vehicle_tmc['TMC_Shap'] = shapley_model.vals_tmc
Vehicle_tmc.to_csv(f'{save_dir}/Vehicle_TMC_1', index=False)
print("[Vehicle] Shapley CSVs saved:")
print("  -", f"{save_dir}/Vehicle_1")
print("  -", f"{save_dir}/Vehicle_test_1")
print("  -", f"{save_dir}/Vehicle_TMC_1")


try:
    df_train = pd.read_csv(f"{save_dir}/Vehicle_1")
    df_test  = pd.read_csv(f"{save_dir}/Vehicle_test_1")
    df_tmc   = pd.read_csv(f"{save_dir}/Vehicle_TMC_1")

    X_train_np = df_train.drop(columns=["target","PerDShap"], errors="ignore").values
    y_train_np = df_train["target"].values
    per_shap   = df_train["PerDShap"].values
    tmc_shap   = df_tmc["TMC_Shap"].values

    X_test_np  = df_test.drop(columns=["target"], errors="ignore").values
    y_test_np  = df_test["target"].values

    out_png_high = os.path.join(plot_dir, "Removal_Vehicle_high_1.png")
    _ = PerformancePlot_Removal(
        vals={"Permutated-Based Data Shapley": per_shap,
              "TMC Shapley": tmc_shap},
        X_train=X_train_np, y_train=y_train_np,
        X_test=X_test_np,   y_test=y_test_np,
        a=0.03, b=0.40, ylim=(70, 105),
        order="descending", random_iters=10, save_path=out_png_high
    )

    out_png_low = os.path.join(plot_dir, "Removal_Vehicle_low_1.png")
    _ = PerformancePlot_Removal(
        vals={"Permutated-Based Data Shapley": per_shap,
              "TMC Shapley": tmc_shap},
        X_train=X_train_np, y_train=y_train_np,
        X_test=X_test_np,   y_test=y_test_np,
        a=0.03, b=0.40, ylim=(70, 105),
        order="ascending", random_iters=10, save_path=out_png_low
    )

    print("[Vehicle] Removal plots saved.")
except Exception as e:
    print("[warn] removal plots failed:", e)


' 5) FashionMNIST '
n_trials     = 1
seed0        = 10231
classes_pool = list(range(10))
k_min, k_max = 2, 3

save_dir = ("/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
            "Data Valuation/DataShapley_Graph/"
            "Datasets/1000pts_I/FashionMNIST")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load FashionMNIST and flatten to vectors, keep labels as ints [0..9]
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()
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

    # Random K-way classification (K in [k_min, k_max])
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # Balanced m/m split with total 2m points across K classes
    m = rng.integers(1000, 1001)  # keep exactly 1000 as in MNIST block
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # Sample per-class indices
    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(n=need_c, random_state=rs + 2001 + c).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # Map labels to {0..K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    # Stratified split: m train / m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    # Standardize -> PCA(20) -> Standardize
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

    # Save test CSV
    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_test_{10+i}", index=False)

    # Permutation-based Data Shapley
    phi = Per_DShap(
        X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_{10+i}", index=False)

    # TMC Shapley via DShap (same settings as MNIST block)
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=350625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_TMC_{10+i}", index=False)

    print(f"[FashionMNIST Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")

# Performance plots (high-valued removal & low-valued removal)
for i, (class_tag, m) in enumerate(zip(trial_tags, trial_ms), start=1):
    base = f"{save_dir}/FashionMNIST_{class_tag}"
    df_tr  = pd.read_csv(f"{base}_{10+i}")
    df_te  = pd.read_csv(f"{base}_test_{10+i}")
    df_tmc = pd.read_csv(f"{base}_TMC_{10+i}")

    X_tr = df_tr.drop(columns=['target', 'PerDShap']).values
    y_tr = df_tr['target'].values
    per_shap = df_tr['PerDShap'].values
    tmc_shap = df_tmc['TMC_Shap'].values

    X_te = df_te.drop(columns=['target']).values
    y_te = df_te['target'].values

    # Remove high-valued first
    PerformancePlot_Removal(
        vals={'Permutated Data Shapley': per_shap, 'TMC Shapley': tmc_shap},
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(80, 105), order='descending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_FashionMNIST_{class_tag}_high_{i}.png"
    )

    # Remove low-valued first
    PerformancePlot_Removal(
        vals={'Permutated Data Shapley': per_shap, 'TMC Shapley': tmc_shap},
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        a=0.03, b=0.4, ylim=(80, 105), order='ascending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_FashionMNIST_{class_tag}_low_{i}.png"
    )

    print(f"[FashionMNIST Trial {i}] Plots saved for tag={class_tag}, m={m}.")




'''
Computation and Comparison of LOO & DShap
'''
'1. Read in dataset with computed DShap values'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr

# =========================================================
# 1. Set working directory
# =========================================================
os.chdir("/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251207")
print("Current working dir:", os.getcwd())

# =========================================================
# 2. Compute LOO scores using logistic regression + val acc
#    LOO_i = U(D) - U(D \ {i})
#    where U(.) is validation accuracy
# =========================================================
def compute_loo_scores(train_df, val_df, dataset_name="dataset"):
    train_df = train_df.copy()
    val_df = val_df.copy()

    # Identify label columns
    label_train = "target"
    label_val = "target"

    # Drop PerDShap from features (if present)
    drop_cols_train = ["PerDShap"]
    drop_cols_train = [c for c in drop_cols_train if c in train_df.columns]

    # Construct feature matrices and labels
    X_train = train_df.drop(columns=drop_cols_train + [label_train])
    y_train = train_df[label_train].values

    X_val = val_df.drop(columns=[label_val])
    y_val = val_df[label_val].values

    print(f"[{dataset_name}] X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # Base model: logistic regression with standardization
    def make_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs")
        )

    # Baseline utility U(D): validation accuracy with full training set
    base_model = make_model()
    base_model.fit(X_train, y_train)
    y_val_pred = base_model.predict(X_val)
    U_full = accuracy_score(y_val, y_val_pred)
    print(f"[{dataset_name}] Baseline validation accuracy U(D) = {U_full:.4f}")

    # LOO loop
    n = X_train.shape[0]
    loo_scores = np.zeros(n, dtype=float)

    print(f"[{dataset_name}] Computing LOO scores for n = {n} points (this may take some time)...")

    for i in range(n):
        # leave-one-out mask
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_tr_i = X_train.iloc[mask]
        y_tr_i = y_train[mask]

        model_i = make_model()
        model_i.fit(X_tr_i, y_tr_i)
        y_val_pred_i = model_i.predict(X_val)
        U_i = accuracy_score(y_val, y_val_pred_i)

        loo_scores[i] = U_full - U_i  # LOO_i = U(D) - U(D\{i})

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{dataset_name}] processed {i+1}/{n}")

    # Attach LOO column to training dataframe
    train_df["LOO_acc"] = loo_scores
    return train_df


# =========================================================
# 3. Plot & correlation between LOO and PerDShap
# =========================================================
def plot_and_corr(df, dataset_name="dataset",
                  loo_col="LOO_acc", shap_col="PerDShap"):

    if loo_col not in df.columns:
        raise ValueError(f"{loo_col} not found in dataframe for {dataset_name}")
    if shap_col not in df.columns:
        raise ValueError(f"{shap_col} not found in dataframe for {dataset_name}")

    x = df[loo_col].values
    y = df[shap_col].values

    # Correlations
    pear_r, pear_p = pearsonr(x, y)
    spear_rho, spear_p = spearmanr(x, y)

    print(f"\n[{dataset_name}] Correlation between {loo_col} and {shap_col}:")
    print(f"  Pearson r   = {pear_r:.4f} (p = {pear_p:.2e})")
    print(f"  Spearman ρ  = {spear_rho:.4f} (p = {spear_p:.2e})")

    # Scatter + fitted line
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6, edgecolors="none")

    # Fit simple linear regression via polyfit
    coef = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coef, x_line)
    plt.plot(x_line, y_line, linewidth=2)

    plt.xlabel("LOO_acc  (U(D) - U(D\\{i\\}))")
    plt.ylabel("PerDShap")
    plt.title(f"{dataset_name}: LOO vs Data Shapley")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =========================================================
# 5. Run for the two datasets: bc and Iris
# =========================================================

# ---- Breast Cancer ----
file_path_bc = "/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251207/bc_1"
file_path_bc_test = "/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251207/bc_test_1"
bc_train = pd.read_csv(file_path_bc)
bc_val   = pd.read_csv(file_path_bc_test)
bc_train_with_loo = compute_loo_scores(bc_train, bc_val, dataset_name="BreastCancer")
# optionally save
bc_train_with_loo.to_csv("bc_1_with_LOO.csv", index=False)
plot_and_corr(bc_train_with_loo, dataset_name="BreastCancer",
              loo_col="LOO_acc", shap_col="PerDShap")


# ---- Iris ----
iris_train = pd.read_csv("Iris_1")
iris_val   = pd.read_csv("Iris_test_1")

iris_train_with_loo = compute_loo_scores(iris_train, iris_val, dataset_name="Iris")
iris_train_with_loo.to_csv("Iris_1_with_LOO.csv", index=False)

plot_and_corr(iris_train_with_loo, dataset_name="Iris",
              loo_col="LOO_acc", shap_col="PerDShap")