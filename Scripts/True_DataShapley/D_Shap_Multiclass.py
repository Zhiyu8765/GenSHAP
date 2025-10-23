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
from sklearn.decomposition import PCA
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
n_trials     = 3
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
    df_te.to_csv(f"{save_dir}/MNIST_{class_tag}_test_{3+i}", index=False)

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True)
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/MNIST_{class_tag}_{3+i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_tag}_TMC_{3+i}", index=False)

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
X_train = pd.DataFrame(pipe.fit_transform(X_train_raw), columns = [f"pc{i+1}" for i in range(20)])
X_test = pd.DataFrame(pipe.transform(X_test_raw), columns = [f"pc{i+1}" for i in range(20)])
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