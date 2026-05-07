''' 0. Import Packages '''
' For Calculation of Ground-Truth Data Shapley Values '
import openml
import math
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
# from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import sparse
import warnings
import random
import tensorflow_datasets as tfds
from tqdm.auto import tqdm as tqdm_auto
from sklearn.exceptions import ConvergenceWarning

' For Calculation of TMC Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
os.chdir('E:/   DataShapley')
# os.chdir('/mnt/data/    /pycharm_project_236/tmp/pycharm_project_236/DataShapley')
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
save_dir = 'E:\  \Datasets\Checked'
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
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

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


' 2) Breast Cancer '
n_trials = 3
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
c_pca = np.random.randint(5, 101)
for i in range(1, n_trials + 1):
    bc0 = bc_0.sample(n=m, random_state=i)
    bc1 = bc_1.sample(n=m, random_state=i)
    bc = pd.concat([bc0, bc1], axis=0).reset_index(drop=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        bc.drop(columns='target'), bc.target, test_size=0.5, stratify=bc.target, random_state=i)
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled = scaler1.transform(X_test_raw)
    # pca = PCA(n_components=20)
    pca = PCA(n_components=c_pca)
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
    df_te.to_csv(f"{save_dir}/MNIST_{class_tag}_test_{i}", index=False)

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True)
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/MNIST_{class_tag}_{i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_tag}_TMC_{i}", index=False)

    print(f"[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")


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


' 5) FashionMNIST '
n_trials     = 3
seed0        = 1112
classes_pool = list(range(10))
k_min, k_max = 2, 4
c_pca = np.random.randint(5, 101)
save_dir = ("E:/   Datasets/Datasets_III")
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
    m = rng.integers(950, 1001)  # keep exactly 1000 as in MNIST block
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

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
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
    df_te.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_test_{i}", index=False)

    # Permutation-based Data Shapley
    phi = Per_DShap(
        X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_{i}", index=False)

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
    df_tmc.to_csv(f"{save_dir}/FashionMNIST_{class_tag}_TMC_{i}", index=False)

    print(f"[FashionMNIST Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")



' 6) CIFAR10'
n_trials     = 3
seed0        = 1112
classes_pool = list(range(10))
k_min, k_max = 2, 5

save_dir = ("E:/   Datasets/Datasets_III")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load CIFAR-10 and flatten images to vectors; labels need ravel() to 1D
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
X_pool = X_train_full.reshape(len(X_train_full), -1)
y_pool = y_train_full.ravel()
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

    # Random K-way classification task (K in [k_min, k_max])
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # Balanced m/m split with total 2m points across K classes
    m = rng.integers(950, 1001)  # keep exactly 1000
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
        idx_c = y_pool[y_pool == c].sample(n=need_c, random_state=rs + 3001 + c).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # Map labels {selected classes} -> {0..K-1}
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

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
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
    df_te.to_csv(f"{save_dir}/CIFAR10_{class_tag}_test_{i}", index=False)

    # Permutation-based Data Shapley
    phi = Per_DShap(
        X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/CIFAR10_{class_tag}_{i}", index=False)

    # TMC Shapley via DShap
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=450625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/CIFAR10_{class_tag}_TMC_{i}", index=False)

    print(f"[CIFAR10 Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")



' 7) CIFAR-100'
n_trials     = 3
seed0        = 1113
classes_pool = list(range(100))
k_min, k_max = 2, 5

save_dir = ("E:/   Datasets/Datasets_III")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load CIFAR-100 and flatten images to vectors
# label_mode='fine' gives 100 fine-grained classes (default)
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
X_pool = X_train_full.reshape(len(X_train_full), -1)
y_pool = y_train_full.ravel()
X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(y_pool, name='target')

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))  # keep same convention

trial_tags = []
trial_ms   = []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # Random K-way classification task (K in [k_min, k_max])
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # Balanced m/m split with total 2m points across K classes
    m = rng.integers(500, 501)  # keep exactly 1000 as before
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # Sample per-class indices from the CIFAR-100 train pool
    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        # each class has exactly 500 train samples, so need_c <= 500 is guaranteed by K>=4
        idx_c = y_pool[y_pool == c].sample(n=need_c, random_state=rs + 4001 + c).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # Map labels {selected_classes} -> {0..K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    # Stratified split: m train / m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    # Standardize -> PCA(20) -> Standardize (same as MNIST / FashionMNIST / CIFAR10)
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
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
    df_te.to_csv(f"{save_dir}/CIFAR100_{class_tag}_test_{i}", index=False)

    # Permutation-based Data Shapley
    phi = Per_DShap(
        X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/CIFAR100_{class_tag}_{i}", index=False)

    # TMC Shapley via DShap
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=550625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/CIFAR100_{class_tag}_TMC_{i}", index=False)

    print(f"[CIFAR100 Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")


'8) Imagenet-A '
image_size = (64, 64)

print("Loading ImageNet-A via TFDS...")
ds_imagenet_a = tfds.load("imagenet_a", split="test", as_supervised=True)

ds_imagenet_a = ds_imagenet_a.map(
    lambda img, lbl: (tf.image.resize(img, image_size), lbl),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

X_list, y_list = [], []
for img, lbl in tfds.as_numpy(ds_imagenet_a):
    X_list.append(img.reshape(-1))
    y_list.append(int(lbl))

X_pool = pd.DataFrame(np.stack(X_list, axis=0))
y_pool = pd.Series(np.array(y_list, dtype=int), name="target")

# 2) Config for Shapley experiments
n_trials     = 3
seed0        = 260123
classes_pool = sorted(y_pool.unique().tolist())  # actual label set in ImageNet-A
k_min, k_max = 2, 5   # number of classes per random task
m_min, m_max = 950, 1001  # target train size per trial (test size = same)

save_dir = ("E:/   Datasets/Datasets_III")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))  # keep same convention as before

trial_tags = []
trial_ms   = []

# 3) Main trials: build tasks, compute PerDShap + TMC, save 3 CSVs
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # Random K-way classification problem
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # Random m per trial
    m = rng.integers(m_min, m_max + 1)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m  # m train + m test
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # Sample per-class indices (with replacement, ImageNet-A is small)
    idx = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need_c,
            random_state=rs + 6001 + c,
            replace=True   # IMPORTANT: allow duplicates due to small dataset
        ).index.tolist()
        idx.extend(idx_c)

    # Build subset DataFrame/Series
    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all = y_pool.iloc[idx].reset_index(drop=True)

    # Map labels {selected_classes} -> {0..K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    # Stratified split: m train / m test
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=m, train_size=m, random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    # Standardisation -> PCA(20) -> standardisation again
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(
        X_train_pca,
        columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])]
    )
    X_test = pd.DataFrame(
        X_test_pca,
        columns=[f'PC{i + 1}' for i in range(X_test_pca.shape[1])]
    )

    # Save test CSV
    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/ImageNetA_{class_tag}_test_{i}", index=False)

    # Permutation-based Data Shapley
    phi = Per_DShap(
        X_train, y_train, X_test, y_test, n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/ImageNetA_{class_tag}_{i}", index=False)

    # TMC Shapley via DShap
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=750625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/ImageNetA_{class_tag}_TMC_{i}", index=False)

    print(f"[ImageNetA Trial {i}] K={K}, classes={selected_classes}, "
          f"m={m}, tag={class_tag}. CSVs saved.")


'9） Caltech101'
image_size = (64, 64)
ds_caltech = tfds.load("caltech101", split="train", as_supervised=True)

# Resize images
ds_caltech = ds_caltech.map(
    lambda img, lbl: (tf.image.resize(img, image_size), lbl),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Convert to numpy
X_list, y_list = [], []
for img, lbl in tfds.as_numpy(ds_caltech):
    X_list.append(img.reshape(-1))
    y_list.append(int(lbl))

X_pool = pd.DataFrame(np.stack(X_list, axis=0))
y_pool = pd.Series(np.array(y_list, dtype=int), name="target")

classes_pool = sorted(y_pool.unique().tolist())
print(f"Loaded Caltech101 with {len(X_pool)} images, {len(classes_pool)} classes.")

n_trials     = 3
seed0        = 260123
k_min, k_max = 2, 5
m_min, m_max = 950, 1001

save_dir = ("E:/   Datasets/Datasets_III")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))

trial_tags = []
trial_ms   = []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # Random K-way classification
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # Random m
    m = rng.integers(m_min, m_max + 1)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # Sample indices (Caltech101 is big enough, but allow replace=True for stability)
    idx = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need_c,
            random_state=rs + 6001 + c,
            replace=True
        ).index.tolist()
        idx.extend(idx_c)

    # Build dataset
    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all = y_pool.iloc[idx].reset_index(drop=True)

    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    # Stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=m,
        train_size=m,
        random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    # Standardize → PCA → Standardize
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])])

    # Save test CSV
    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/Caltech101_{class_tag}_test_{i}.csv", index=False)

    # ============================================================
    # Permutation Data Shapley
    # ============================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )

    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/Caltech101_{class_tag}_{i}.csv", index=False)

    # ============================================================
    # TMC Shapley
    # ============================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family='logistic',
        metric='accuracy',
        overwrite=True,
        directory='./temp',
        seed=750625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/Caltech101_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Caltech101 Trial {i}] K={K}, classes={selected_classes}, m={m}")



' 10) STL10'
image_size = (64, 64)

print("Loading STL10 via TFDS...")
ds_stl10 = tfds.load("stl10", split="train", as_supervised=True)

# Resize images to 64x64
ds_stl10 = ds_stl10.map(
    lambda img, lbl: (tf.image.resize(img, image_size), lbl),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Convert to numpy arrays
X_list, y_list = [], []
for img, lbl in tfds.as_numpy(ds_stl10):
    X_list.append(img.reshape(-1))      # flatten HWC -> vector
    y_list.append(int(lbl))             # int label 0–9

X_pool = pd.DataFrame(np.stack(X_list, axis=0))
y_pool = pd.Series(np.array(y_list, dtype=int), name="target")

classes_pool = sorted(y_pool.unique().tolist())
print(f"Loaded STL10 with {len(X_pool)} images, {len(classes_pool)} classes: {classes_pool}")


n_trials     = 3
seed0        = 260123
k_min, k_max = 2, 5       # number of classes per random task
m_min, m_max = 950, 1001  # m train, m test per trial

save_dir = "E:/   Datasets/Datasets_III"
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))  # pad to fixed length as before

trial_tags = []
trial_ms   = []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # ----- Random K-way classification problem -----
    K = rng.integers(k_min, k_max + 1)
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # ----- Random m -----
    m = rng.integers(m_min, m_max + 1)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m  # m train + m test
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # ----- Sample indices per class -----
    idx = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need_c,
            random_state=rs + 6001 + c,
            replace=True   # safe; STL10 train has 5k samples
        ).index.tolist()
        idx.extend(idx_c)

    # Build subset
    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all = y_pool.iloc[idx].reset_index(drop=True)

    # Map original labels -> {0,...,K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map)

    # ----- Stratified split: m train / m test -----
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=m, train_size=m, random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[test_idx].reset_index(drop=True)

    # ----- Standardisation -> PCA(20) -> standardisation again -----
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca = PCA(n_components=np.random.randint(5, 101), random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(
        X_train_pca,
        columns=[f'PC{i + 1}' for i in range(X_train_pca.shape[1])]
    )
    X_test = pd.DataFrame(
        X_test_pca,
        columns=[f'PC{i + 1}' for i in range(X_test_pca.shape[1])]
    )

    # ----- Save test CSV -----
    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/STL10_{class_tag}_test_{i}.csv", index=False)

    # ============================================================
    # Permutation-based Data Shapley
    # ============================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, multi_class='auto'),
        random_state=0, progress=True
    )

    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/STL10_{class_tag}_{i}.csv", index=False)

    # ============================================================
    # TMC Shapley via DShap
    # ============================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family='logistic',
        metric='accuracy',
        overwrite=True,
        directory='./temp_stl10',
        seed=750625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/STL10_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[STL10 Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")


'''
Datasets in OpenML
'''
' 11) Letter'
n_trials     = 3
seed0        = 260126
k_min, k_max = 2, 5   # credit-g is binary; keep K=2
save_dir = ("E:/   Datasets/Datasets_III")
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---- Load credit-g from OpenML ----
from sklearn.datasets import fetch_openml

# X_pool, y_pool = fetch_openml(name="eeg-eye-state", version="active", as_frame=True, return_X_y=True, parser="auto")
# X_pool, y_pool = fetch_openml(name="mozilla4", version="active", as_frame=True, return_X_y=True, parser="auto")
X_pool, y_pool = fetch_openml(name="letter", version="active", as_frame=True, return_X_y=True, parser="auto")

X_pool = pd.DataFrame(X_pool)
# y_pool = pd.Series(y_pool.astype(int), name="target")  # already 0/1
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")
print("Class counts:", y_pool.value_counts().sort_index())

# Map string labels -> {0,1} deterministically
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

# classes_pool = sorted(y_pool.unique().tolist())  # should be [0,1]
classes_pool = sorted(y_pool.unique().tolist())  # [0, ..., 25]

def tag_from_classes(cls_list):
    cls_list = sorted(cls_list)
    s = ''.join(str(c) for c in cls_list)
    return s + 'x' * (5 - len(cls_list))

trial_tags = []
trial_ms   = []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    K = np.random.randint(2, 6)
    selected_classes = np.random.choice(classes_pool, size=K, replace=False)
    class_tag = tag_from_classes(selected_classes)

    # mimic your MNIST design: m in [1000,1000]
    m = rng.integers(900,1001)
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"Not enough samples in class {c}: need {need_c}, but only {len(grp)} available."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    # ---- Preprocess: one-hot cats + scale + PCA(20) + scale (like MNIST block) ----
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_enc)
    X_test_scaled  = scaler1.transform(X_test_enc)

    # pca = PCA(n_components=20, random_state=rs)
    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f'PC{k + 1}' for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f'PC{k + 1}' for k in range(X_test_pca.shape[1])])

    # ---- Save TEST ----
    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/letter_{class_tag}_test_{i}", index=False)

    # ---- Permutation-based Data Shapley (PerDShap) ----
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200),
        random_state=0,
        progress=True
    )
    df_tr = X_train.copy()
    df_tr['target']   = y_train.values
    df_tr['PerDShap'] = phi
    df_tr.to_csv(f"{save_dir}/letter_{class_tag}_{i}", index=False)

    # ---- TMC Shapley ----
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test), sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625 + i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=['PerDShap']).copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/letter_{class_tag}_TMC_{i}", index=False)

    print(f"[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}. CSVs saved.")


' 14) nomao'
DATASET_NAME = "nomao"
n_trials     = 3
seed0        = 260126

# For nomao it is binary; K will be forced to 2 automatically
k_min, k_max = 2, 5

# Keep these modest for runtime; increase if you want
m_min, m_max = 900, 1001   # train size = m, test size = m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False  # set True if you want removal plots and have PerformancePlot_Removal

# =========================================================
# 2) Load OpenML dataset: nomao
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)

# Map labels -> integer codes (binary strings -> {0,1})
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

# Drop any rows with missing labels (safety)
mask = ~y_pool.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool = y_pool.loc[mask].reset_index(drop=True)

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

def tag_from_classes(cls_list):
    # unambiguous tag
    return "_".join(map(str, cls_list))

trial_tags = []
trial_ms   = []

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # nomao is binary; enforce K=2
    if n_classes_total < 2:
        raise ValueError("Dataset appears to have <2 classes after loading.")
    K = 2
    selected_classes = classes_pool[:]  # typically [0,1]
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points, split evenly across 2 classes
    total_needed = 2 * m
    per_class_counts = [total_needed // 2, total_needed - (total_needed // 2)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0].union(idx_all[1])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # Re-map labels to {0,1} (still good practice)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess (nomao has missing values)
    #    - numeric: impute(median) -> scale
    #    - categorical: impute(most_frequent) -> one-hot
    #    - then scale -> PCA -> scale (MNIST style)
    # =========================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    transformers = []
    if len(num_cols) > 0:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            num_cols
        ))
    if len(cat_cols) > 0:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        ))

    if len(transformers) == 0:
        raise ValueError("No usable columns found after dtype inspection.")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_enc)
    X_test_scaled  = scaler1.transform(X_test_enc)

    # SAFE PCA dim cap (nomao can still end up with small n_features depending on preprocessing)
    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    # You must have Per_DShap defined earlier (same as your MNIST section)
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'15) bank-marketing'
DATASET_NAME = "bank-marketing"
n_trials     = 3
seed0        = 260127

# bank-marketing is binary; K will be forced to 2 automatically
k_min, k_max = 2, 5

# Adjust m for runtime; bank-marketing is large so 1000/1000 is fine
m_min, m_max = 950, 1000     # train size = m, test size = m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False  # True if you want removal plots and have PerformancePlot_Removal

# =========================================================
# 2) Load OpenML dataset: bank-marketing
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)

# Map labels -> integer codes (typically 'yes'/'no' -> {0,1})
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

# Drop any rows with missing labels (safety)
mask = ~y_pool.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool = y_pool.loc[mask].reset_index(drop=True)

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

def tag_from_classes(cls_list):
    return "_".join(map(str, cls_list))

trial_tags = []
trial_ms   = []

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # bank-marketing is binary; enforce K=2
    if n_classes_total < 2:
        raise ValueError("Dataset appears to have <2 classes after loading.")
    K = 2
    selected_classes = classes_pool[:]  # typically [0,1]
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points, split (almost) evenly across 2 classes
    total_needed = 2 * m
    per_class_counts = [total_needed // 2, total_needed - (total_needed // 2)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0].union(idx_all[1])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # Re-map labels to {0,1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess (bank-marketing has categoricals + possible missing values)
    #    - numeric: impute(median) -> scale
    #    - categorical: impute(most_frequent) -> one-hot
    #    - then scale -> PCA -> scale (MNIST style)
    # =========================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    transformers = []
    if len(num_cols) > 0:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            num_cols
        ))
    if len(cat_cols) > 0:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        ))

    if len(transformers) == 0:
        raise ValueError("No usable columns found after dtype inspection.")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_enc)
    X_test_scaled  = scaler1.transform(X_test_enc)

    # SAFE PCA dim cap
    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'16) pendigits'
DATASET_NAME = "pendigits"   # <-- switched here
n_trials     = 3
seed0        = 260127

# Multi-class (10 digits). Sample K classes per trial like MNIST.
k_min, k_max = 2, 5

# Train size = m, test size = m
m_min, m_max = 950, 1000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False  # True if you want removal plots and have PerformancePlot_Removal

def tag_from_classes(cls_list):
    # Unambiguous even for multi-digit, though pendigits is 0..9 anyway
    return "_".join(map(str, sorted(cls_list)))

trial_tags = []
trial_ms   = []

# =========================================================
# 2) Load OpenML dataset: pendigits
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")  # map to 0..C-1

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K and classes (like MNIST)
    K = int(rng.integers(k_min, min(k_max, n_classes_total) + 1))
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points distributed (almost) evenly across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # re-map selected classes to 0..K-1 (important for clean multiclass)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale -> PCA(<=20) -> scale (MNIST style)
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'17) artificial character'
DATASET_NAME = "artificial-characters"   # <-- switched here
n_trials     = 3
seed0        = 260127

# Multi-class. Sample K classes per trial like MNIST.
k_min, k_max = 2, 5

# Train size = m, test size = m
m_min, m_max = 800, 900

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False  # True if you want removal plots and have PerformancePlot_Removal

def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))

trial_tags = []
trial_ms   = []

# =========================================================
# 2) Load OpenML dataset: artificial-characters
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")  # map to 0..C-1

# Drop missing labels (rare, but safe)
mask = ~y_pool.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool = y_pool.loc[mask].reset_index(drop=True)

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K and classes (like MNIST)
    K = int(rng.integers(k_min, min(k_max, n_classes_total) + 1))
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points distributed (almost) evenly across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # re-map selected classes to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: impute -> scale -> PCA(<=20) -> scale
    # =========================================================
    # (Most OpenML "artificial-characters" variants are numeric; this is a safe pipeline.)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale1", StandardScaler()),
    ])

    X_train_scaled = pipe.fit_transform(X_train_raw.values)
    X_test_scaled  = pipe.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


' 18) har'
DATASET_NAME = "har"
n_trials     = 3
seed0        = 1215

# HAR has 6 classes
k_min, k_max = 2, 5

m_min, m_max = 900, 1000   # train = m, test = m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False

def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))

trial_tags = []
trial_ms   = []

# =========================================================
# 2) Load OpenML dataset: HAR
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes
    K = int(rng.integers(k_min, min(k_max, n_classes_total) + 1))
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample 2m points evenly across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # re-map selected classes to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=m, train_size=m, random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale -> PCA -> scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'18) gas-drift'
DATASET_NAME = "gas-drift"
n_trials     = 3
seed0        = 260128

# gas-drift is multi-class (several gases). Sample K classes per trial.
k_min, k_max = 2, 5

m_min, m_max = 900, 1000   # train = m, test = m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False

def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))

trial_tags = []
trial_ms   = []

# =========================================================
# 2) Load OpenML dataset: gas-drift
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")  # map to 0..C-1

# drop missing labels (safe)
mask = ~y_pool.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool = y_pool.loc[mask].reset_index(drop=True)

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K and classes (like MNIST)
    K = int(rng.integers(k_min, min(k_max, n_classes_total) + 1))
    selected_classes = rng.choice(classes_pool, size=K, replace=False)
    selected_classes = sorted(selected_classes.tolist())
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points distributed evenly across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]
        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(grp)}."
            )
        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # re-map selected classes to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: impute -> scale -> PCA(<=20) -> scale
    # =========================================================
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale1", StandardScaler()),
    ])

    X_train_scaled = pipe.fit_transform(X_train_raw.values)
    X_test_scaled  = pipe.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'19) connect-4'
DATASET_NAME = "connect-4"
# OpenML shows two active dataset IDs for connect-4
OPENML_IDS = [1591, 40668]

n_trials = 3
seed0    = 260128
k_min, k_max = 2, 3
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))


# =========================================================
# 2) Load dataset via openml-python (ROBUST)
# =========================================================
def load_connect4_via_openml(ids):
    last_err = None
    for data_id in ids:
        try:
            ds = openml.datasets.get_dataset(data_id)
            target_name = ds.default_target_attribute
            X, y, cat_indicator, feat_names = ds.get_data(
                target=target_name,
                dataset_format="dataframe"   # <- reliable: gives DataFrame + Series
            )

            # y should be strings like 'win'/'loss'/'draw' (or similar)
            y = pd.Series(y, name="target").reset_index(drop=True)
            X = X.reset_index(drop=True)

            # sanity
            uniq = y.unique()
            if len(uniq) != 3:
                raise RuntimeError(
                    f"Loaded data_id={data_id} but got {len(uniq)} target labels: {uniq[:10]}"
                )

            print(f"Loaded connect-4 from OpenML data_id={data_id}, target='{target_name}'")
            return X, y

        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load connect-4 from OpenML ids={ids}. Last error: {last_err}")


X_pool, y_pool_raw = load_connect4_via_openml(OPENML_IDS)

# Map labels -> integer codes (0..2)
y_pool = pd.Series(pd.Categorical(y_pool_raw).codes, name="target")

print("Dataset:", DATASET_NAME)
print("X_pool shape:", X_pool.shape)
print("Raw target labels:", pd.unique(y_pool_raw))
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
n_classes_total = len(classes_pool)  # should be 3


# =========================================================
# 3) Trials
# =========================================================
trial_tags, trial_ms = [], []

for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K and classes
    K = int(rng.integers(k_min, min(k_max, n_classes_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    # sample total 2m points near-uniform over K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need_c > len(cand):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need_c}, but only {len(cand)}."
            )
        idx_all.append(rng.choice(cand, size=need_c, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # remap selected classes -> 0..K-1 (like your MNIST block)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess (connect-4 is categorical-heavy)
    #   - impute -> one-hot (cats) + impute/scale (nums) -> scale -> PCA -> scale
    # =========================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    # Column-wise preprocessing
    transformers = []

    if len(num_cols) > 0:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                # IMPORTANT: may receive sparse slices in some sklearn/openml configs
                ("scale", StandardScaler(with_mean=False)),
            ]),
            num_cols
        ))

    if len(cat_cols) > 0:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),  # outputs sparse by default
            ]),
            cat_cols
        ))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        # Optional: keep sparse output (default). We handle sparsity explicitly below.
        # sparse_threshold=1.0
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc = pre.transform(X_test_raw)

    # Outer scaling: output is usually sparse due to one-hot
    scaler1 = StandardScaler(with_mean=False)
    X_train_scaled = scaler1.fit_transform(X_train_enc)
    X_test_scaled = scaler1.transform(X_test_enc)

    # PCA requires dense input
    if sparse.issparse(X_train_scaled):
        X_train_scaled_dense = X_train_scaled.toarray()
        X_test_scaled_dense = X_test_scaled.toarray()
    else:
        X_train_scaled_dense = np.asarray(X_train_scaled)
        X_test_scaled_dense = np.asarray(X_test_scaled)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled_dense.shape[0], X_train_scaled_dense.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled_dense)
    X_test_pca = pca.transform(X_test_scaled_dense)

    # Final scaling on PCs (dense)
    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k + 1}" for k in range(X_train_pca.shape[1])])
    X_test = pd.DataFrame(X_test_pca, columns=[f"PC{k + 1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) PerDShap (your existing function)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'20) covertype'
DATASET_NAME = "covertype"
n_trials     = 3
seed0        = 260129

# covertype has 7 classes → allow partial class subsets
k_min, k_max = 2, 5

m_min, m_max = 900, 1000   # train = m, test = m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 2) Load dataset (robust)
# =========================================================
X_pool, y_pool = fetch_openml(
    name="covertype",
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

print("Dataset:", DATASET_NAME)
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)  # should be 7

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes
    K = int(rng.integers(k_min, min(k_max, C_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    m = int(rng.integers(m_min, m_max + 1))

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}")

    # total 2m samples, balanced across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need > len(cand):
            raise ValueError(f"Not enough samples in class {c}")
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all = y_pool.iloc[idx].reset_index(drop=True)

    # re-map labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all.map(label_map)

    # stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=m, test_size=m, random_state=rs
    )
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale → PCA → scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) Permutation Data Shapley
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Finished.")


'21) CovPokElec'
DATASET_NAME = "CovPokElec"
n_trials     = 3
seed0        = 260129

# multi-class subset selection per trial
k_min, k_max = 2, 5

m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # produce "123" instead of "1_2_3"
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 2) Load dataset from OpenML
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)

# map labels -> 0..C-1 (works for strings or numbers)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")
keep = y_pool.isin(range(7))
X_pool = X_pool.loc[keep].reset_index(drop=True)
y_pool = y_pool.loc[keep].reset_index(drop=True)

# refresh class list
classes_pool = sorted(y_pool.unique().tolist())  # now [0,1,2,3,4,5,6]
C_total = len(classes_pool)
print("Filtered class counts:\n", y_pool.value_counts().sort_index())

print("Dataset:", DATASET_NAME)
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
print("Number of classes:", C_total)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes for this trial
    K = int(rng.integers(k_min, min(k_max, C_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    # choose m
    m = int(rng.integers(m_min, m_max + 1))

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")

    # sample 2m points balanced across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need > len(cand):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need}, but only {len(cand)}."
            )
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # re-map selected classes to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale -> PCA(<=20) -> scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) Permutation Data Shapley (PerDShap)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'22) Pokerhand'
DATASET_NAME = "pokerhand"
n_trials     = 3
seed0        = 260129

k_min, k_max = 2, 5
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # "123" instead of "1_2_3"
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 2) Load dataset from OpenML
# =========================================================
X_pool, y_pool = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)

# map labels -> 0..C-1
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")
keep = y_pool.isin(range(7))
X_pool = X_pool.loc[keep].reset_index(drop=True)
y_pool = y_pool.loc[keep].reset_index(drop=True)

# refresh class list
classes_pool = sorted(y_pool.unique().tolist())  # now [0,1,2,3,4,5,6]
C_total = len(classes_pool)
print("Filtered class counts:\n", y_pool.value_counts().sort_index())

print("Dataset:", DATASET_NAME)
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
print("Number of classes:", C_total)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes
    K = int(rng.integers(k_min, min(k_max, C_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    m = int(rng.integers(m_min, m_max + 1))

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")

    # sample 2m points balanced across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need > len(cand):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need}, but only {len(cand)}."
            )
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # remap selected classes -> 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale -> PCA(<=20) -> scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) Permutation Data Shapley (PerDShap)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'23) LED(50000)'
from sklearn.datasets import fetch_openml
DATASET_NAME_CANDIDATES = ["LED(50000)", "LED_50000", "LED50000", "LED"]

n_trials     = 3
seed0        = 260130

k_min, k_max = 2, 5
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # "123" instead of "1_2_3"
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 2) Load dataset from OpenML (robust name fallback)
# =========================================================
last_err = None
X_pool = y_pool = None
used_name = None

for nm in DATASET_NAME_CANDIDATES:
    try:
        X_pool, y_pool = fetch_openml(
            name=nm,
            version="active",
            as_frame=True,
            return_X_y=True,
            parser="auto"
        )
        used_name = nm
        break
    except Exception as e:
        last_err = e

if used_name is None:
    raise RuntimeError(
        f"Failed to fetch OpenML dataset with names {DATASET_NAME_CANDIDATES}. "
        f"Last error: {last_err}"
    )

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_pool).codes, name="target")

print(f"Dataset: {used_name}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
print("Number of classes:", C_total)

# =========================================================
# 3) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes
    K = int(rng.integers(k_min, min(k_max, C_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    m = int(rng.integers(m_min, m_max + 1))

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")

    # sample 2m points balanced across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need > len(cand):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need}, but only {len(cand)}."
            )
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # remap selected classes -> 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 4) Preprocess: scale -> PCA(<=20) -> scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 5) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/LED_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 6) Permutation Data Shapley (PerDShap)
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/LED_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 7) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/LED_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")

'24) RANDOMRBF_0_0'
DATASET_NAME = "RandomRBF_0_0"
n_trials     = 3
seed0        = 260130

k_min, k_max = 2, 5
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # "123" instead of "1_2_3"
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 2) Load dataset (robust to sparse ARFF)
# =========================================================
def load_openml_safe(name: str):
    # Try frame first; some OpenML datasets forbid as_frame=True (sparse ARFF)
    try:
        X, y = fetch_openml(
            name=name, version="active",
            as_frame=True, return_X_y=True, parser="auto"
        )
        X_df = pd.DataFrame(X)
        y_arr = np.asarray(y).ravel()
        return X_df, y_arr, "as_frame=True"
    except ValueError as e:
        msg = str(e)
        if "Sparse ARFF datasets cannot be loaded with as_frame=True" not in msg:
            raise
        bunch = fetch_openml(
            name=name, version="active",
            as_frame=False, return_X_y=False, parser="auto"
        )
        X = bunch.data
        y = bunch.target
        # keep X as numpy/sparse; we will only densify after sampling
        y_arr = np.asarray(y).ravel()
        return X, y_arr, "as_frame=False"

X_pool_raw, y_raw, load_mode = load_openml_safe(DATASET_NAME)

# Map labels -> 0..C-1 (works for strings/numbers)
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

# Print basic info (without forcing dense on X)
if isinstance(X_pool_raw, pd.DataFrame):
    n_samples, n_features = X_pool_raw.shape
else:
    n_samples, n_features = X_pool_raw.shape

print(f"Dataset: {DATASET_NAME} ({load_mode})")
print("X shape:", (n_samples, n_features))
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
print("Number of classes:", C_total)

# =========================================================
# 3) Helper: take rows from X_pool_raw without densifying everything
# =========================================================
def take_rows(X, idx):
    """Return X[idx] as a dense numpy array (only for the sampled subset)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].to_numpy()
    # numpy or sparse matrix
    X_sub = X[idx]
    if sparse.issparse(X_sub):
        return X_sub.toarray()
    return np.asarray(X_sub)

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K classes for this trial
    K = int(rng.integers(k_min, min(k_max, C_total) + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    m = int(rng.integers(m_min, m_max + 1))
    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")

    # sample 2m points balanced across K classes
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    y_vals = y_pool.values
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = np.where(y_vals == c)[0]
        if need > len(cand):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: need {need}, but only {len(cand)}."
            )
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # Build the sampled dataset as dense (only 2m rows)
    X_all = take_rows(X_pool_raw, idx)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # remap selected classes -> 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all[tr_idx]
    X_test_raw  = X_all[te_idx]
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 5) Preprocess: scale -> PCA(<=20) -> scale
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 6) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 7) PerDShap
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 8) TMC Shapley (DShap)
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'25) jannis'
DATASET_NAME = "jannis"
n_trials     = 3
seed0        = 260130

k_min, k_max = 2, 4         # jannis has multiple classes
m_min, m_max = 900, 1000     # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

MAKE_PLOTS = False  # set True if you want removal plots

def tag_from_classes(cls):
    # 1_2_3 -> 123
    return "".join(map(str, sorted(cls)))

# =========================================================
# 3) Load dataset (numeric, safe)
# =========================================================
X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose subset of classes
    K = rng.integers(k_min, min(k_max, C_total) + 1)
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False))
    class_tag = tag_from_classes(selected_classes)

    m = rng.integers(m_min, m_max + 1)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}")

    # balanced sampling: 2m points
    total_needed = 2 * m
    base = total_needed // K
    rem  = total_needed % K
    per_class = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        cand = y_pool[y_pool == c].index
        if per_class[j] > len(cand):
            raise ValueError(f"Class {c} too small")
        idx_all.append(
            rng.choice(cand, size=per_class[j], replace=False)
        )

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # remap labels -> 0..K-1
    y_all = y_all_raw.map({c: j for j, c in enumerate(selected_classes)})

    # stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=m, test_size=m, random_state=rs
    )
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test  = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 5) Scale + PCA (PCA ONLY)
    # =====================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 6) Save TEST
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/jannis_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 7) Permutation Data Shapley
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=200, solver="lbfgs", multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/jannis_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 8) TMC Shapley
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/jannis_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs.")


' Shuttle'
REPO_DIR = r"E:/   DataShapley_Graph/Datasets_III"
if os.path.isdir(REPO_DIR):
    os.chdir(REPO_DIR)
    sys.path.append(os.getcwd())

DATASET_NAME = "shuttle"
n_trials     = 3
seed0        = 260202
k_min, k_max = 2, 5
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls):
    # "1_2_3" -> "123" (no underscores)
    return "".join(map(str, sorted(cls)))

# =========================================================
# 4) Load OpenML shuttle
# =========================================================
X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

# =========================================================
# 5) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    m = int(rng.integers(m_min, m_max + 1))
    total_needed = 2 * m

    # --- choose K and classes, but ONLY from classes with enough samples ---
    # We need roughly total_needed/K per class; worst-case need is ceil(total_needed/K).
    # We'll try K from high to low until feasible.
    counts = y_pool.value_counts().sort_index()
    all_classes = counts.index.to_list()

    feasible = False
    for K_try in range(min(k_max, len(all_classes)), k_min - 1, -1):
        # random K within [k_min, k_max] but must be feasible
        # we’ll sample K later; first compute minimum required per class if balanced
        base = total_needed // K_try
        rem  = total_needed % K_try
        # max required for any class under our balancing scheme
        need_max = base + (1 if rem > 0 else 0)

        eligible = [c for c in all_classes if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            # now pick K: random in [k_min, min(k_max, len(eligible))]
            K = int(rng.integers(k_min, min(k_max, len(eligible)) + 1))
            # recompute needs for this K
            base = total_needed // K
            rem  = total_needed % K
            need_max = base + (1 if rem > 0 else 0)
            eligible = [c for c in all_classes if counts.loc[c] >= need_max]
            if len(eligible) >= K:
                selected_classes = sorted(rng.choice(eligible, size=K, replace=False).tolist())
                feasible = True
                break

    if not feasible:
        raise RuntimeError(
            f"Could not find enough classes with >= needed samples for m={m}. "
            f"Try reducing m_min/m_max or k_max."
        )

    class_tag = tag_from_classes(selected_classes)

    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # balanced per-class counts
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(n=need, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # remap selected classes -> 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=m, train_size=m, random_state=rs)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 6) Standardize + PCA (PCA ONLY)
    # =====================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled  = scaler1.transform(X_test_raw)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 7) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/shuttle_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 8) PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/shuttle_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 9) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/shuttle_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'spoken-arabic-digit'
DATASET_NAME = "spoken-arabic-digit"

n_trials     = 3
seed0        = 260202

k_min, k_max = 2, 5
m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls):
    return "".join(map(str, sorted(cls)))  # e.g., [0,1,2] -> "012"

# =========================================================
# 4) Load OpenML dataset (robust)
# =========================================================
def load_openml_safe(name):
    try:
        X, y = fetch_openml(
            name=name, version="active",
            as_frame=True, return_X_y=True, parser="auto"
        )
        return pd.DataFrame(X), np.asarray(y).ravel(), "as_frame=True"
    except ValueError as e:
        if "Sparse ARFF datasets cannot be loaded with as_frame=True" not in str(e):
            raise
        bunch = fetch_openml(
            name=name, version="active",
            as_frame=False, return_X_y=False, parser="auto"
        )
        X = bunch.data
        y = np.asarray(bunch.target).ravel()
        if sparse.issparse(X):
            X = X.toarray()  # safe because we only later sample 2m anyway
        return pd.DataFrame(np.asarray(X)), y, "as_frame=False"

X_pool, y_raw, mode = load_openml_safe(DATASET_NAME)

# map labels -> codes
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME} ({mode})")
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)

# =========================================================
# 5) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # choose K and classes (only if enough samples per class for balanced 2m)
    m = int(rng.integers(m_min, m_max + 1))
    total_needed = 2 * m

    # pick a random K in [k_min, k_max], but ensure feasibility
    for _try in range(50):
        K = int(rng.integers(k_min, min(k_max, C_total) + 1))
        base = total_needed // K
        rem  = total_needed % K
        need_max = base + (1 if rem > 0 else 0)

        counts = y_pool.value_counts()
        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K:
            selected_classes = sorted(rng.choice(eligible, size=K, replace=False).tolist())
            break
    else:
        raise RuntimeError("Could not find feasible class subset; reduce m or k_max.")

    class_tag = tag_from_classes(selected_classes)
    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # balanced sampling: 2m points across K classes
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # remap labels -> 0..K-1
    y_all = y_all_raw.map({c: j for j, c in enumerate(selected_classes)}).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 6) Standardize + PCA (PCA ONLY)
    # =====================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 7) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/spoken-arabic-digit_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 8) PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/spoken-arabic-digit_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 9) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/spoken-arabic-digit_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'dilbert'
DATASET_NAME = "dilbert"

n_trials = 3
seed0    = 260202

# dilbert is binary in most OpenML versions -> fix K=2
k_min, k_max = 2, 5

m_min, m_max = 900, 1000  # train=m, test=m

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # "0_1" -> "01"
    return "".join(map(str, sorted(cls_list)))

def densify(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)

# =========================================================
# 4) Load OpenML dilbert BY NAME
# =========================================================
# Use as_frame='auto' to avoid sparse ARFF issues while still preferring DataFrame
X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)

# X_pool may be DataFrame, ndarray, or sparse
if isinstance(X_pool, pd.DataFrame):
    X_df = X_pool.copy()
else:
    # if sparse -> keep sparse for now; wrap later after sampling+PCA
    X_df = X_pool

y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X type:", type(X_df))
if hasattr(X_df, "shape"):
    print("X shape:", X_df.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
if len(classes_pool) < 2:
    raise RuntimeError("Need at least 2 classes for Shapley utility (accuracy).")

# =========================================================
# 5) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    m = int(rng.integers(m_min, m_max + 1))
    total_needed = 2 * m

    # binary
    K = 2
    selected_classes = classes_pool[:2]
    class_tag = tag_from_classes(selected_classes)

    # balanced sampling across the two classes
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index.to_numpy()
        if need > len(cand):
            raise ValueError(f"[Trial {i}] Not enough samples in class {c}: need {need}, only {len(cand)}.")
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # take rows
    if isinstance(X_df, pd.DataFrame):
        X_all = X_df.iloc[idx].reset_index(drop=True)
    else:
        # ndarray or sparse matrix
        X_all = X_df[idx]

    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)

    # remap labels -> 0..1
    label_map = {selected_classes[0]: 0, selected_classes[1]: 1}
    y_all = y_all_raw.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(np.zeros(len(y_all)), y_all))

    # split X
    if isinstance(X_all, pd.DataFrame):
        X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
        X_train_mat = X_train_raw.values
        X_test_mat  = X_test_raw.values
    else:
        X_train_mat = X_all[tr_idx]
        X_test_mat  = X_all[te_idx]

    y_train = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test  = y_all.iloc[te_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")
    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 6) Scale + PCA (PCA ONLY)
    # =====================================================
    # Use with_mean=False to be safe if X is sparse.
    scaler1 = StandardScaler(with_mean=False)
    X_train_sc = scaler1.fit_transform(X_train_mat)
    X_test_sc  = scaler1.transform(X_test_mat)

    # enforce PCA-only: densify after scaling
    X_train_dense = densify(X_train_sc)
    X_test_dense  = densify(X_test_sc)

    pca_dim = min(np.random.randint(5, 101), X_train_dense.shape[0], X_train_dense.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_dense)
    X_test_pca  = pca.transform(X_test_dense)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 7) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/dilbert_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 8) PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(max_iter=300, solver="lbfgs"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/dilbert_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 9) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/dilbert_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'kuzushiji-MNIST'
DATASET_NAME = "Kuzushiji-MNIST"
DATASET_TAG  = "Kuzushiji-MNIST"

n_trials = 3
seed0    = 260203

# KMNIST has 10 classes; choose K classes each trial
k_min, k_max = 2, 5

PCA_TARGET = np.random.randint(5, 101)
N_PERM     = 5000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

def tag_from_classes(cls_list):
    # e.g. [0,1,2] -> "012"
    return "".join(map(str, sorted(cls_list)))

# =========================================================
# 3) Load Kuzushiji-MNIST by NAME
# =========================================================
X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)

# y -> codes 0..C-1
y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

# X: ensure 2D numeric matrix (KMNIST is typically numeric already)
if isinstance(X_pool, pd.DataFrame):
    X_mat = X_pool.to_numpy()
else:
    X_mat = X_pool

if sparse.issparse(X_mat):
    # keep sparse until we densify after scaling (but KMNIST usually dense)
    X_mat = X_mat
else:
    X_mat = np.asarray(X_mat)

if X_mat.ndim != 2:
    raise ValueError(f"Unexpected X shape: {X_mat.shape}")

print(f"Dataset: {DATASET_NAME}")
print("X shape:", X_mat.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for accuracy-based utility.")

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # m random in [900, 1000] using Python random (seeded per trial)
    random.seed(rs)
    m = random.randint(900, 1000)
    total_needed = 2 * m

    # Choose feasible K/classes (exclude tiny classes automatically)
    counts = y_pool.value_counts()
    selected_classes = None
    K = None

    for _try in range(150):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))
        base = total_needed // K_try
        rem  = total_needed % K_try
        need_max = base + (1 if rem > 0 else 0)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError("Could not find feasible class subset. Reduce m or k_max.")

    class_tag = tag_from_classes(selected_classes)
    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # Balanced sampling for 2m points across K classes
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index.to_numpy()
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # Slice and remap labels -> 0..K-1
    X_all = X_mat[idx]
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(np.zeros(len(y_all)), y_all))

    X_train_raw = X_all[tr_idx]
    X_test_raw  = X_all[te_idx]
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 5) Intended preprocessing: Standardize -> PCA -> Standardize
    # =====================================================
    # (1) Standardize features (sparse-safe)
    scaler1 = StandardScaler(with_mean=False)
    X_train_sc = scaler1.fit_transform(X_train_raw)
    X_test_sc  = scaler1.transform(X_test_raw)

    # PCA requires dense -> densify AFTER scaling (only 2m subset, safe)
    X_train_dense = X_train_sc.toarray() if sparse.issparse(X_train_sc) else np.asarray(X_train_sc)
    X_test_dense  = X_test_sc.toarray()  if sparse.issparse(X_test_sc)  else np.asarray(X_test_sc)

    # (2) PCA to <= 20 (bounded by n_samples/n_features)
    pca_dim = min(PCA_TARGET, X_train_dense.shape[0], X_train_dense.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_dense)
    X_test_pca  = pca.transform(X_test_dense)

    # (3) Standardize PCs
    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 6) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 7) PerDShap (Permutation-based)
    # =====================================================
    X_np = X_train.values
    y_np = y_train.values
    Xv   = X_test.values
    yv   = y_test.values
    N    = len(X_np)
    baseline = 0.0

    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    phi = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=int)

    try:
        from tqdm import tqdm
        perm_iter = tqdm(range(N_PERM), desc="Permutation Data Shapley", unit="perm")
    except Exception:
        perm_iter = range(N_PERM)

    rng_perm = np.random.default_rng(0)
    for _ in perm_iter:
        perm = rng_perm.permutation(N)
        prev = baseline
        for j in range(1, N + 1):
            vj = utility(perm[:j])
            ii = perm[j - 1]
            mc = vj - prev
            cnt[ii] += 1
            phi[ii] += (mc - phi[ii]) / cnt[ii]
            prev = vj

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 8) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")



'Devnagari-Script'
DATASET_NAME = "Devnagari-Script"
DATASET_TAG  = "Devnagari-Script"
n_trials = 3
seed0    = 260203
k_min, k_max = 2, 5
PCA_TARGET = np.random.randint(5, 101)
N_PERM     = 5000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)

# X -> numeric 2D array
if isinstance(X_pool, pd.DataFrame):
    X_mat = X_pool.to_numpy()
else:
    X_mat = np.asarray(X_pool)

if X_mat.ndim != 2:
    raise ValueError(f"Unexpected X shape: {X_mat.shape}")

# y -> integer codes 0..C-1
y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X shape:", X_mat.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for accuracy-based utility.")

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # m random in [900, 1000] using Python random (seeded per trial)
    random.seed(rs)
    m = random.randint(900, 1000)
    total_needed = 2 * m

    # pick feasible K/classes (exclude tiny classes automatically)
    counts = y_pool.value_counts()
    selected_classes = None
    K = None

    for _try in range(200):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))
        base = total_needed // K_try
        rem  = total_needed % K_try
        need_max = base + (1 if rem > 0 else 0)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError("Could not find feasible class subset. Reduce m or k_max.")

    # naming tag: e.g., [1,3,10] -> "1310" (no separators)
    class_tag = "".join(map(str, selected_classes))

    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # Balanced per-class sampling for 2m points
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index.to_numpy()
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # Slice X/y and remap labels -> 0..K-1
    X_all = X_mat[idx]
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(np.zeros(len(y_all)), y_all))

    X_train_raw = X_all[tr_idx]
    X_test_raw  = X_all[te_idx]
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 5) Intended preprocessing: Standardize -> PCA -> Standardize
    # =====================================================
    scaler1 = StandardScaler()
    X_train_sc = scaler1.fit_transform(X_train_raw)
    X_test_sc  = scaler1.transform(X_test_raw)

    pca_dim = min(PCA_TARGET, X_train_sc.shape[0], X_train_sc.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 6) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 7) PerDShap (Permutation-based)
    # =====================================================
    X_np = X_train.values
    y_np = y_train.values
    Xv   = X_test.values
    yv   = y_test.values
    N    = len(X_np)
    baseline = 0.0

    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    phi = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=int)

    try:
        from tqdm import tqdm
        perm_iter = tqdm(range(N_PERM), desc="Permutation Data Shapley", unit="perm")
    except Exception:
        perm_iter = range(N_PERM)

    rng_perm = np.random.default_rng(0)
    for _ in perm_iter:
        perm = rng_perm.permutation(N)
        prev = baseline
        for j in range(1, N + 1):
            vj = utility(perm[:j])
            ii = perm[j - 1]
            mc = vj - prev
            cnt[ii] += 1
            phi[ii] += (mc - phi[ii]) / cnt[ii]
            prev = vj

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 8) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'Volkert'
DATASET_TAG = "volkert"
n_trials = 3
seed0    = 114

k_min, k_max = 2, 5
PCA_TARGET = np.random.randint(5, 101)
N_PERM     = 5000

save_dir = r"E:/   Datasets/Datasets_III"
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# 3) Load dataset by NAME (inline try a few variants)
# =========================================================
name_candidates = ["volkert", "Volkert", "VOLKERT"]
last_err = None
X_pool = None
y_raw = None

for nm in name_candidates:
    try:
        X_pool, y_raw = fetch_openml(
            name=nm,
            version="active",
            as_frame="auto",
            return_X_y=True,
            parser="auto"
        )
        print(f"Loaded OpenML dataset by name='{nm}'")
        break
    except Exception as e:
        last_err = e

if X_pool is None:
    raise RuntimeError(f"Failed to fetch OpenML dataset with names {name_candidates}. Last error: {last_err}")

# Ensure DataFrame for consistent categorical handling
if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset tag: {DATASET_TAG}")
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for accuracy-based utility.")

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # m random in [900, 1000] using Python random (seeded per trial)
    random.seed(rs)
    m = random.randint(900, 1000)
    total_needed = 2 * m

    # Choose feasible K/classes (exclude tiny classes automatically)
    counts = y_pool.value_counts()
    selected_classes = None
    K = None

    for _try in range(200):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))
        base = total_needed // K_try
        rem  = total_needed % K_try
        need_max = base + (1 if rem > 0 else 0)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError("Could not find feasible class subset. Reduce m or k_max.")

    # naming tag like "012" (no underscores)
    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # Balanced sampling for 2m points across K classes
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index.to_numpy()
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # Slice and remap labels -> 0..K-1
    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 5) Intended preprocessing: Encode -> Standardize -> PCA -> Standardize
    # =====================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    # (1) Standardize features (sparse-safe)
    scaler1 = StandardScaler(with_mean=False)
    X_train_sc = scaler1.fit_transform(X_train_enc)
    X_test_sc  = scaler1.transform(X_test_enc)

    # densify AFTER scaling (only 2m subset)
    X_train_dense = X_train_sc.toarray() if sparse.issparse(X_train_sc) else np.asarray(X_train_sc)
    X_test_dense  = X_test_sc.toarray()  if sparse.issparse(X_test_sc)  else np.asarray(X_test_sc)

    # (2) PCA to <= 20
    pca_dim = min(PCA_TARGET, X_train_dense.shape[0], X_train_dense.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_dense)
    X_test_pca  = pca.transform(X_test_dense)

    # (3) Standardize PCs
    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 6) Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 7) PerDShap (Permutation-based)
    # =====================================================
    X_np = X_train.values
    y_np = y_train.values
    Xv   = X_test.values
    yv   = y_test.values
    N    = len(X_np)
    baseline = 0.0

    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    phi = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=int)

    try:
        from tqdm import tqdm
        perm_iter = tqdm(range(N_PERM), desc="Permutation Data Shapley", unit="perm")
    except Exception:
        perm_iter = range(N_PERM)

    rng_perm = np.random.default_rng(0)
    for _ in perm_iter:
        perm = rng_perm.permutation(N)
        prev = baseline
        for j in range(1, N + 1):
            vj = utility(perm[:j])
            ii = perm[j - 1]
            mc = vj - prev
            cnt[ii] += 1
            phi[ii] += (mc - phi[ii]) / cnt[ii]
            prev = vj

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 8) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_TAG}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs (test / PerDShap / TMC).")


'hls4ml_lhc_jets_hlf'
DATASET_NAME = "hls4ml_lhc_jets_hlf"
FILE_PREFIX  = "hls4ml"
n_trials = 3
seed0    = 260204
k_min, k_max = 2, 5
PCA_TARGET = np.random.randint(5, 101)
N_PERM     = 5000
# m random in [900, 1000]
m_lo, m_hi = 900, 1000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# 3) Load dataset by NAME (no helper wrapper)
# =========================================================
X_pool, y_raw = fetch_openml(
    name=DATASET_NAME,
    version="active",
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)

# Ensure DataFrame
if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for accuracy-based utility.")

# =========================================================
# 4) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    random.seed(rs)
    m = random.randint(m_lo, m_hi)
    total_needed = 2 * m

    # Choose feasible K/classes (exclude tiny classes automatically)
    counts = y_pool.value_counts()
    selected_classes = None
    K = None

    for _try in range(200):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))
        base = total_needed // K_try
        rem  = total_needed % K_try
        need_max = base + (1 if rem > 0 else 0)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError("Could not find feasible class subset. Reduce m or k_max.")

    # class_tag without underscores, e.g. [0,1,2] -> "012"
    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {i}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # Balanced sampling for 2m points across K classes
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx_all = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        cand = y_pool[y_pool == c].index.to_numpy()
        idx_all.append(rng.choice(cand, size=need, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    # Slice and remap labels -> 0..K-1
    X_all = X_pool.iloc[idx].reset_index(drop=True)
    y_all_raw = y_pool.iloc[idx].reset_index(drop=True)
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # 5) Preprocess: OneHot(if needed) -> Std -> PCA -> Std
    # =====================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    # Standardize features (sparse-safe)
    scaler1 = StandardScaler(with_mean=False)
    X_train_sc = scaler1.fit_transform(X_train_enc)
    X_test_sc  = scaler1.transform(X_test_enc)

    # PCA needs dense -> densify after scaling (only 2m subset)
    X_train_dense = X_train_sc.toarray() if sparse.issparse(X_train_sc) else np.asarray(X_train_sc)
    X_test_dense  = X_test_sc.toarray()  if sparse.issparse(X_test_sc)  else np.asarray(X_test_sc)

    pca_dim = min(PCA_TARGET, X_train_dense.shape[0], X_train_dense.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_dense)
    X_test_pca  = pca.transform(X_test_dense)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # 6) Save TEST CSV (prefix renamed)
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{i}.csv", index=False)

    # =====================================================
    # 7) PerDShap (Permutation-based)
    # =====================================================
    X_np = X_train.values
    y_np = y_train.values
    Xv   = X_test.values
    yv   = y_test.values
    N    = len(X_np)
    baseline = 0.0

    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    phi = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=int)

    try:
        from tqdm import tqdm
        perm_iter = tqdm(range(N_PERM), desc="Permutation Data Shapley", unit="perm")
    except Exception:
        perm_iter = range(N_PERM)

    rng_perm = np.random.default_rng(0)
    for _ in perm_iter:
        perm = rng_perm.permutation(N)
        prev = baseline
        for j in range(1, N + 1):
            vj = utility(perm[:j])
            ii = perm[j - 1]
            mc = vj - prev
            cnt[ii] += 1
            phi[ii] += (mc - phi[ii]) / cnt[ii]
            prev = vj

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{i}.csv", index=False)

    # =====================================================
    # 8) TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs: "
          f"{FILE_PREFIX}_{class_tag}_test_{i}.csv, "
          f"{FILE_PREFIX}_{class_tag}_{i}.csv, "
          f"{FILE_PREFIX}_{class_tag}_TMC_{i}.csv")

'autos'
DATASET_NAME = "autos"
FILE_PREFIX  = "autos"

n_trials = 3
seed0    = 260204

k_min, k_max = 2, 5
m_min, m_max = 950, 1000
PCA_TARGET   = np.random.randint(5, 101)
N_PERM       = 5000

save_dir = r"E:/   Datasets/Datasets_III"   # EDIT
os.makedirs(save_dir, exist_ok=True)

# ---- load autos by name ----
X_pool, y_raw = fetch_openml(
    data_id=45080,
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)

if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

y_raw = np.asarray(y_raw).ravel()

# If target is numeric-continuous, this dataset might not be a classification task in your setting.
# For OpenML autos, classification tasks exist (e.g., predicting 'symboling').  [oai_citation:1‡openml.org](https://www.openml.org/search?order=asc&q=+run_flow.flow_id%3A17950&sort=nr_of_likes&type=task&utm_source=chatgpt.com)
# Here we treat the provided OpenML target as categorical labels:
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
counts = y_pool.value_counts()

def tag_from_classes(cls_list):
    # e.g., [0,1,3] -> "013"
    return "".join(map(str, sorted(cls_list)))

for trial in range(1, n_trials + 1):
    rs = seed0 + trial
    rng = np.random.default_rng(rs)
    random.seed(rs)

    # pick K and classes that can support 2m points WITHOUT replacement
    m = random.randint(m_min, m_max)
    total_needed = 2 * m

    selected_classes = None
    K = None

    for _ in range(500):
        K_try = int(rng.integers(k_min, min(k_max, len(classes_pool)) + 1))

        base = total_needed // K_try
        rem  = total_needed % K_try
        per_class_need = [base + (1 if j < rem else 0) for j in range(K_try)]
        need_max = max(per_class_need)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError(
            f"Could not find feasible (K,m) for autos without replacement. "
            f"Try smaller m_max (currently {m_max})."
        )

    class_tag = tag_from_classes(selected_classes)
    print(f"\n[Trial {trial}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # balanced sampling across selected classes (NO replacement)
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(n=need, random_state=rs + 1007 + int(c), replace=False).index
        idx.extend(idx_c.tolist())

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # remap labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # ---- preprocess: (impute+onehot) -> std -> PCA -> std ----
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    X_train_enc = pre.fit_transform(X_train_raw)
    X_test_enc  = pre.transform(X_test_raw)

    scaler1 = StandardScaler()
    X_train_sc = scaler1.fit_transform(X_train_enc)
    X_test_sc  = scaler1.transform(X_test_enc)

    pca_dim = min(PCA_TARGET, X_train_sc.shape[0], X_train_sc.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # ---- save TEST ----
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{trial}.csv", index=False)

    # ---- PerDShap ----
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=N_PERM,
        model_builder=lambda: LogisticRegression(max_iter=300, multi_class="auto"),
        random_state=0,
        progress=True
    )
    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{trial}.csv", index=False)

    # ---- TMC Shapley ----
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + trial,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv", index=False)

    print(f"[autos Trial {trial}] Saved 3 CSVs for tag={class_tag}.")

'Credit_Score'
DATASET_NAME = "dataset_credit_score"   # OpenML name  [oai_citation:1‡openml.org](https://openml.org/d/46537?utm_source=chatgpt.com)
FILE_PREFIX  = "Credit_Score"

n_trials = 3
seed0    = 260204

k_min, k_max = 2, 5
PCA_TARGET   = np.random.randint(5, 101)
N_PERM       = 5000

# pick m based on typical size; adjust if you want
m_min, m_max = 900, 1000

save_dir = r"E:/   Datasets/Datasets_III"   # EDIT
os.makedirs(save_dir, exist_ok=True)
# -----------------------------
# Load from OpenML
# -----------------------------
X_pool, y_raw = fetch_openml(
    data_id = 46506,
    version="active",
    as_frame="auto",
    return_X_y=True,
    parser="auto"
)
y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print("X shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
counts = y_pool.value_counts()

# =========================================================
# Trials
# =========================================================
for trial in range(1, n_trials + 1):
    rs = seed0 + trial
    rng = np.random.default_rng(rs)

    # m by random package
    random.seed(rs)
    m = random.randint(m_min, m_max)
    total_needed = 2 * m

    # pick feasible K and classes (no replacement)
    selected_classes = None
    K = None

    for _ in range(500):
        K_try = int(rng.integers(k_min, min(k_max, len(classes_pool)) + 1))

        base = total_needed // K_try
        rem  = total_needed % K_try
        per_class_need = [base + (1 if j < rem else 0) for j in range(K_try)]
        need_max = max(per_class_need)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError(f"Cannot find feasible class subset for m={m}. Try smaller m_max.")

    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {trial}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # balanced sampling (NO replacement)
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need, random_state=rs + 1007 + int(c), replace=False
        ).index
        idx.extend(idx_c.tolist())

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # remap labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[te_idx].reset_index(drop=True)

    # =====================================================
    # (1) Force Occupation + Payment_Behaviour as categorical
    # =====================================================
    force_cat = ["Occupation", "Payment_Behaviour"]
    for col in force_cat:
        if col in X_train_raw.columns:
            X_train_raw[col] = X_train_raw[col].astype("category")
            X_test_raw[col]  = X_test_raw[col].astype("category")

    # =====================================================
    # (2) Impute ALL NaN as 0 (both train/test)
    # =====================================================
    X_train_raw = X_train_raw.fillna(0)
    X_test_raw  = X_test_raw.fillna(0)

    # =====================================================
    # (3) Split cat/num columns
    #     Categorical = object/category/bool + the forced columns
    # =====================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # ensure forced cols included if present
    for col in force_cat:
        if col in X_train_raw.columns and col not in cat_cols:
            cat_cols.append(col)
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    # =====================================================
    # (4) Normalize ALL numerical variables (StandardScaler)
    # =====================================================
    scaler_num = StandardScaler()
    Xtr_num = scaler_num.fit_transform(X_train_raw[num_cols].astype(float)) if len(num_cols) > 0 else np.zeros((len(X_train_raw), 0))
    Xte_num = scaler_num.transform(X_test_raw[num_cols].astype(float))      if len(num_cols) > 0 else np.zeros((len(X_test_raw), 0))

    # =====================================================
    # (5) One-hot encode categorical variables (after NaN->0)
    # =====================================================
    # (Note: NaN already imputed as 0, so 0 becomes a valid category token)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xtr_cat = ohe.fit_transform(X_train_raw[cat_cols].astype(str)) if len(cat_cols) > 0 else np.zeros((len(X_train_raw), 0))
    Xte_cat = ohe.transform(X_test_raw[cat_cols].astype(str))      if len(cat_cols) > 0 else np.zeros((len(X_test_raw), 0))

    # Concatenate [normalized numeric | one-hot categorical]
    Xtr_feat = np.hstack([Xtr_num, Xtr_cat])
    Xte_feat = np.hstack([Xte_num, Xte_cat])

    # =====================================================
    # (6) PCA
    # =====================================================
    pca_dim = min(PCA_TARGET, Xtr_feat.shape[0], Xtr_feat.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    Xtr_pca = pca.fit_transform(Xtr_feat)
    Xte_pca = pca.transform(Xte_feat)

    # =====================================================
    # (7) Normalize again (on PCA features)
    # =====================================================
    scaler2 = StandardScaler()
    Xtr_final = scaler2.fit_transform(Xtr_pca)
    Xte_final = scaler2.transform(Xte_pca)

    X_train = pd.DataFrame(Xtr_final, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(Xte_final, columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{trial}.csv", index=False)

    # =====================================================
    # PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=N_PERM,
        model_builder=lambda: LogisticRegression(max_iter=300, multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{trial}.csv", index=False)

    # =====================================================
    # TMC Shapley
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + trial,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv", index=False)

    print(f"[Trial {trial}] Saved 3 CSVs: "
          f"{FILE_PREFIX}_{class_tag}_test_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv")


'SDSS17'
DATA_ID     = 46955
FILE_PREFIX = "SDSS17"

n_trials = 3
seed0    = 260205

k_min, k_max = 2, 5
m_min, m_max = 900, 1000
PCA_TARGET   = np.random.randint(5, 101)
N_PERM       = 5000

save_dir = r"E:/   Datasets/Datasets_III"   # EDIT
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Load dataset by data_id
# -----------------------------
X_pool, y_raw = fetch_openml(
    data_id=DATA_ID,
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

# (1) Drop "plate" (minimal + safe)
if "plate" in X_pool.columns:
    X_pool = X_pool.drop(columns=["plate"])

# y -> categorical codes
y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Loaded SDSS17 from OpenML data_id={DATA_ID}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
counts = y_pool.value_counts()
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for classification/Shapley.")

# =========================================================
# Trials
# =========================================================
for trial in range(1, n_trials + 1):
    rs = seed0 + trial
    rng = np.random.default_rng(rs)

    # m random using random package
    random.seed(rs)
    m = random.randint(m_min, m_max)
    total_needed = 2 * m

    # pick feasible K and classes (NO replacement)
    selected_classes = None
    K = None

    for _ in range(500):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))

        base = total_needed // K_try
        rem  = total_needed % K_try
        per_class_need = [base + (1 if j < rem else 0) for j in range(K_try)]
        need_max = max(per_class_need)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError(
            f"Cannot find feasible class subset for m={m} without replacement. "
            f"Try smaller m_max."
        )

    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {trial}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # balanced sampling across chosen classes (NO replacement)
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need,
            random_state=rs + 1007 + int(c),
            replace=False
        ).index
        idx.extend(idx_c.tolist())

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # remap labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())


    # =====================================================
    # (2) Force fiber_ID as categorical + one-hot
    # =====================================================
    if "fiber_ID" in X_train_raw.columns:
        # force categorical via string conversion (robust even if numeric IDs)
        X_train_raw["fiber_ID"] = X_train_raw["fiber_ID"].astype(str)
        X_test_raw["fiber_ID"]  = X_test_raw["fiber_ID"].astype(str)

    # Identify categorical columns (non-numeric) + fiber_ID forced
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if "fiber_ID" in X_train_raw.columns and "fiber_ID" not in cat_cols:
        cat_cols.append("fiber_ID")

    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    # =====================================================
    # Normalize numeric variables
    # =====================================================
    if len(num_cols) > 0:
        scaler_num = StandardScaler()
        Xtr_num = scaler_num.fit_transform(X_train_raw[num_cols].astype(float))
        Xte_num = scaler_num.transform(X_test_raw[num_cols].astype(float))
    else:
        Xtr_num = np.zeros((len(X_train_raw), 0))
        Xte_num = np.zeros((len(X_test_raw), 0))

    # =====================================================
    # One-hot categorical variables (includes fiber_ID)
    # =====================================================
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr_cat = ohe.fit_transform(X_train_raw[cat_cols].astype(str))
        Xte_cat = ohe.transform(X_test_raw[cat_cols].astype(str))
    else:
        Xtr_cat = np.zeros((len(X_train_raw), 0))
        Xte_cat = np.zeros((len(X_test_raw), 0))

    # Concatenate
    Xtr_feat = np.hstack([Xtr_num, Xtr_cat])
    Xte_feat = np.hstack([Xte_num, Xte_cat])

    # =====================================================
    # PCA
    # =====================================================
    pca_dim = min(PCA_TARGET, Xtr_feat.shape[0], Xtr_feat.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    Xtr_pca = pca.fit_transform(Xtr_feat)
    Xte_pca = pca.transform(Xte_feat)

    # =====================================================
    # Normalize again (on PCA features)
    # =====================================================
    scaler2 = StandardScaler()
    Xtr_final = scaler2.fit_transform(Xtr_pca)
    Xte_final = scaler2.transform(Xte_pca)

    X_train = pd.DataFrame(Xtr_final, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(Xte_final, columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{trial}.csv", index=False)

    # =====================================================
    # PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=N_PERM,
        model_builder=lambda: LogisticRegression(max_iter=300, multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{trial}.csv", index=False)

    # =====================================================
    # TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + trial,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv", index=False)

    print(f"[Trial {trial}] Saved 3 CSVs: "
          f"{FILE_PREFIX}_{class_tag}_test_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv")


'WBCAtt'
DATA_ID     = 46676   # WBCAtt on OpenML  [oai_citation:1‡OpenML](https://www.openml.org/search?id=46676&sort=runs&status=active&type=data&utm_source=chatgpt.com)
FILE_PREFIX = "WBCAtt"

n_trials = 3
seed0    = 260205

k_min, k_max = 2, 5
m_min, m_max = 900, 1000
PCA_TARGET   = np.random.randint(5, 101)
N_PERM       = 5000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
X_pool, y_raw = fetch_openml(
    data_id=DATA_ID,
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Loaded WBCAtt from OpenML data_id={DATA_ID}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
counts = y_pool.value_counts()
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for classification/Shapley.")

# =========================================================
# Trials
# =========================================================
for trial in range(1, n_trials + 1):
    rs = seed0 + trial
    rng = np.random.default_rng(rs)

    # m random using Python random package
    random.seed(rs)
    m = random.randint(m_min, m_max)
    total_needed = 2 * m

    # -----------------------------------------------------
    # Choose K and classes that can support sampling
    # WITHOUT replacement (avoids duplicated rows).
    # -----------------------------------------------------
    selected_classes = None
    K = None

    for _ in range(500):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))

        base = total_needed // K_try
        rem  = total_needed % K_try
        per_class_need = [base + (1 if j < rem else 0) for j in range(K_try)]
        need_max = max(per_class_need)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError(
            f"Cannot find feasible class subset for m={m} without replacement. "
            f"Try smaller m_max."
        )

    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {trial}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # -----------------------------------------------------
    # Sample exactly 2m points balanced across the K classes
    # -----------------------------------------------------
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need,
            random_state=rs + 1007 + int(c),
            replace=False
        ).index
        idx.extend(idx_c.tolist())

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # remap labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map).astype(int)

    # stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # Impute missing values safely:
    #   numeric -> 0
    #   categorical/object/bool -> "0"
    # =====================================================
    cat_cols0 = X_train_raw.select_dtypes(include=["category", "object", "bool"]).columns
    num_cols0 = X_train_raw.columns.difference(cat_cols0)

    if len(num_cols0) > 0:
        X_train_raw[num_cols0] = X_train_raw[num_cols0].fillna(0)
        X_test_raw[num_cols0]  = X_test_raw[num_cols0].fillna(0)

    if len(cat_cols0) > 0:
        X_train_raw[cat_cols0] = X_train_raw[cat_cols0].astype(str).fillna("0")
        X_test_raw[cat_cols0]  = X_test_raw[cat_cols0].astype(str).fillna("0")

    # =====================================================
    # Split categorical / numeric again after casting
    # =====================================================
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    # =====================================================
    # Normalize numeric
    # =====================================================
    if len(num_cols) > 0:
        scaler_num = StandardScaler()
        Xtr_num = scaler_num.fit_transform(X_train_raw[num_cols].astype(float))
        Xte_num = scaler_num.transform(X_test_raw[num_cols].astype(float))
    else:
        Xtr_num = np.zeros((len(X_train_raw), 0))
        Xte_num = np.zeros((len(X_test_raw), 0))

    # =====================================================
    # One-hot categorical
    # =====================================================
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr_cat = ohe.fit_transform(X_train_raw[cat_cols].astype(str))
        Xte_cat = ohe.transform(X_test_raw[cat_cols].astype(str))
    else:
        Xtr_cat = np.zeros((len(X_train_raw), 0))
        Xte_cat = np.zeros((len(X_test_raw), 0))

    # Concatenate features
    Xtr_feat = np.hstack([Xtr_num, Xtr_cat])
    Xte_feat = np.hstack([Xte_num, Xte_cat])

    # =====================================================
    # PCA
    # =====================================================
    pca_dim = min(PCA_TARGET, Xtr_feat.shape[0], Xtr_feat.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    Xtr_pca = pca.fit_transform(Xtr_feat)
    Xte_pca = pca.transform(Xte_feat)

    # =====================================================
    # Normalize again (on PCA features)
    # =====================================================
    scaler2 = StandardScaler()
    Xtr_final = scaler2.fit_transform(Xtr_pca)
    Xte_final = scaler2.transform(Xte_pca)

    X_train = pd.DataFrame(Xtr_final, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(Xte_final,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{trial}.csv", index=False)

    # =====================================================
    # PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=N_PERM,
        model_builder=lambda: LogisticRegression(max_iter=300, multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{trial}.csv", index=False)

    # =====================================================
    # TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + trial,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv", index=False)

    print(f"[Trial {trial}] Saved 3 CSVs: "
          f"{FILE_PREFIX}_{class_tag}_test_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv")


'Financial-Risk-Assessment'
DATA_ID     = 46481
FILE_PREFIX = "FinancialRisk"

n_trials = 3
seed0    = 260205

k_min, k_max = 2, 5
m_min, m_max = 900, 1000
PCA_TARGET   = np.random.randint(5, 101)
N_PERM       = 5000

save_dir = r"E:/   Datasets/Datasets_III"  # EDIT
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Load dataset by ID
# -----------------------------
X_pool, y_raw = fetch_openml(
    data_id=DATA_ID,
    as_frame=True,
    return_X_y=True,
    parser="auto"
)

if not isinstance(X_pool, pd.DataFrame):
    X_pool = pd.DataFrame(X_pool)

y_raw = np.asarray(y_raw).ravel()
y_pool = pd.Series(pd.Categorical(y_raw).codes, name="target")

print(f"Loaded Financial-Risk-Assessment from OpenML data_id={DATA_ID}")
print("Raw X shape:", X_pool.shape)
print("Raw class counts:\n", y_pool.value_counts().sort_index())

# =========================================================
# Preprocessing requirements (apply BEFORE sampling/splitting)
# 1) remove rows with missing values
# 2) drop "city"
# =========================================================
df_all = X_pool.copy()
df_all["target"] = y_pool.values

# Drop "city" (if present)
if "city" in df_all.columns:
    df_all = df_all.drop(columns=["city"])

# Drop rows with ANY missing values (includes target if ever missing)
before_n = len(df_all)
df_all = df_all.dropna(axis=0).reset_index(drop=True)
after_n = len(df_all)
print(f"Dropped rows with missing values: {before_n - after_n} removed, {after_n} remaining.")

X_pool = df_all.drop(columns=["target"])
y_pool = df_all["target"].astype(int)

print("Post-clean X shape:", X_pool.shape)
print("Post-clean class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())
counts = y_pool.value_counts()
C_total = len(classes_pool)
if C_total < 2:
    raise RuntimeError("Need at least 2 classes for classification/Shapley.")

# =========================================================
# Categorical / Numerical column definitions (user-specified)
# =========================================================
cat_cols_spec = [
    "gender", "education_level", "marital_status", "loan_purpose",
    "employment_status", "payment_history", "state", "country",
    "previous_defaults", "marital_status_change"
]

# Keep only those that actually exist in the dataset
cat_cols = [c for c in cat_cols_spec if c in X_pool.columns]
missing_cat = [c for c in cat_cols_spec if c not in X_pool.columns]
if len(missing_cat) > 0:
    print("Warning: these specified categorical columns were not found and will be ignored:", missing_cat)

# All other columns are numerical (as requested)
num_cols = [c for c in X_pool.columns if c not in cat_cols]

# =========================================================
# Trials
# =========================================================
for trial in range(1, n_trials + 1):
    rs = seed0 + trial
    rng = np.random.default_rng(rs)

    random.seed(rs)
    m = random.randint(m_min, m_max)
    total_needed = 2 * m

    # -----------------------------------------------------
    # Choose K and class subset feasible WITHOUT replacement
    # -----------------------------------------------------
    selected_classes = None
    K = None

    for _ in range(500):
        K_try = int(rng.integers(k_min, min(k_max, C_total) + 1))

        base = total_needed // K_try
        rem  = total_needed % K_try
        per_class_need = [base + (1 if j < rem else 0) for j in range(K_try)]
        need_max = max(per_class_need)

        eligible = [c for c in classes_pool if counts.loc[c] >= need_max]
        if len(eligible) >= K_try:
            selected_classes = sorted(rng.choice(eligible, size=K_try, replace=False).tolist())
            K = K_try
            break

    if selected_classes is None:
        raise RuntimeError(
            f"Cannot find feasible class subset for m={m} without replacement. "
            f"Try smaller m_max."
        )

    class_tag = "".join(map(str, selected_classes))
    print(f"\n[Trial {trial}] m={m}, K={K}, classes={selected_classes}, tag={class_tag}")

    # -----------------------------------------------------
    # Sample exactly 2m points balanced across the K classes
    # -----------------------------------------------------
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    idx = []
    for j, c in enumerate(selected_classes):
        need = per_class_counts[j]
        idx_c = y_pool[y_pool == c].sample(
            n=need,
            random_state=rs + 1007 + int(c),
            replace=False
        ).index
        idx.extend(idx_c.tolist())

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True)

    # Remap labels to 0..K-1
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all_mapped = y_all.map(label_map).astype(int)

    # Stratified split: m train, m test
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, test_size=m, random_state=rs)
    tr_idx, te_idx = next(sss.split(X_all, y_all_mapped))

    X_train_raw = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[te_idx].reset_index(drop=True)
    y_train     = y_all_mapped.iloc[tr_idx].reset_index(drop=True)
    y_test      = y_all_mapped.iloc[te_idx].reset_index(drop=True)

    print("Train counts:\n", y_train.value_counts().sort_index())
    print("Test  counts:\n", y_test.value_counts().sort_index())

    # =====================================================
    # Preprocess: one-hot(cat) + standardize(num) -> PCA -> standardize
    # =====================================================
    # One-hot categorical (treat as string tokens)
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr_cat = ohe.fit_transform(X_train_raw[cat_cols].astype(str))
        Xte_cat = ohe.transform(X_test_raw[cat_cols].astype(str))
    else:
        Xtr_cat = np.zeros((len(X_train_raw), 0))
        Xte_cat = np.zeros((len(X_test_raw), 0))

    # Standardize numerical
    if len(num_cols) > 0:
        scaler_num = StandardScaler()
        Xtr_num = scaler_num.fit_transform(X_train_raw[num_cols].astype(float))
        Xte_num = scaler_num.transform(X_test_raw[num_cols].astype(float))
    else:
        Xtr_num = np.zeros((len(X_train_raw), 0))
        Xte_num = np.zeros((len(X_test_raw), 0))

    # Concatenate
    Xtr_feat = np.hstack([Xtr_num, Xtr_cat])
    Xte_feat = np.hstack([Xte_num, Xte_cat])

    # PCA
    pca_dim = min(PCA_TARGET, Xtr_feat.shape[0], Xtr_feat.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    Xtr_pca = pca.fit_transform(Xtr_feat)
    Xte_pca = pca.transform(Xte_feat)

    # Standardize again (on PCA features)
    scaler2 = StandardScaler()
    Xtr_final = scaler2.fit_transform(Xtr_pca)
    Xte_final = scaler2.transform(Xte_pca)

    X_train = pd.DataFrame(Xtr_final, columns=[f"PC{k+1}" for k in range(pca_dim)])
    X_test  = pd.DataFrame(Xte_final,  columns=[f"PC{k+1}" for k in range(pca_dim)])

    # =====================================================
    # Save TEST CSV
    # =====================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_test_{trial}.csv", index=False)

    # =====================================================
    # PerDShap
    # =====================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=N_PERM,
        model_builder=lambda: LogisticRegression(max_iter=300, multi_class="auto"),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"]   = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_{trial}.csv", index=False)

    # =====================================================
    # TMC Shapley (DShap)
    # =====================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + trial,
        max_iter=100,
        err=0.005,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv", index=False)

    print(f"[Trial {trial}] Saved 3 CSVs: "
          f"{FILE_PREFIX}_{class_tag}_test_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_{trial}.csv, "
          f"{FILE_PREFIX}_{class_tag}_TMC_{trial}.csv")

' 19) Dry Bean'
from ucimlrepo import fetch_ucirepo
DATASET_NAME = "drybean"
n_trials     = 3
seed0        = 260326
k_min, k_max = 2, 5          # random K-way task from the 7 bean classes
m_min, m_max = 900, 1000     # train size = m, test size = m

save_dir = r"E:/   Datasets/Datasets_III"   # EDIT if needed
plot_dir = os.path.join(save_dir, "PerformancePlots")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))

# =========================================================
# 1) Load Dry Bean from UCI
# =========================================================
dry_bean = fetch_ucirepo(id=602)

X_pool = pd.DataFrame(dry_bean.data.features).copy()
y_pool_raw = pd.Series(dry_bean.data.targets.squeeze(), name="target")

# remove rows with missing labels if any
mask = ~y_pool_raw.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool_raw = y_pool_raw.loc[mask].reset_index(drop=True)

# map original bean names -> integer codes
y_pool = pd.Series(pd.Categorical(y_pool_raw).codes, name="target")

print(f"Dataset: {DATASET_NAME}")
print("X_pool shape:", X_pool.shape)
print("Class counts:\n", y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())

trial_tags = []
trial_ms   = []

# =========================================================
# 2) Trials
# =========================================================
for i in range(1, n_trials + 1):
    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # ----- Random K-way classification task -----
    K = int(rng.integers(k_min, k_max + 1))
    selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
    class_tag = tag_from_classes(selected_classes)

    # ----- Choose m -----
    m = int(rng.integers(m_min, m_max + 1))
    trial_tags.append(class_tag)
    trial_ms.append(m)

    total_needed = 2 * m   # m for train, m for test
    base = total_needed // K
    rem  = total_needed % K
    per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

    # ----- Sample selected classes -----
    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]

        if need_c > len(grp):
            raise ValueError(
                f"[Trial {i}] Not enough samples in class {c}: "
                f"need {need_c}, but only {len(grp)} available."
            )

        idx_c = grp.sample(n=need_c, random_state=rs + 1007 + int(c)).index
        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # Re-map selected classes to {0, ..., K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # ----- Stratified split: m train / m test -----
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=m,
        train_size=m,
        random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    print(f"\n[Trial {i}] K={K}, classes={selected_classes}, m={m}, tag={class_tag}")
    print("Train class counts:\n", y_train.value_counts().sort_index())
    print("Test  class counts:\n", y_test.value_counts().sort_index())

    # =========================================================
    # 3) Preprocess: scale -> PCA -> scale
    #    Dry Bean is numeric tabular data, so no one-hot needed.
    # =========================================================
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw.values)
    X_test_scaled  = scaler1.transform(X_test_raw.values)

    pca_dim = min(np.random.randint(5, 101), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=rs)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca  = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(X_train_pca, columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])])
    X_test  = pd.DataFrame(X_test_pca,  columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])])

    # =========================================================
    # 4) Save TEST CSV
    # =========================================================
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    df_te.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv", index=False)

    # =========================================================
    # 5) Permutation-based Data Shapley
    # =========================================================
    phi = Per_DShap(
        X_train, y_train, X_test, y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi
    df_tr.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv", index=False)

    # =========================================================
    # 6) TMC Shapley via DShap
    # =========================================================
    shapley_model = DShap(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory="./temp",
        seed=250625 + i,
        max_iter=100,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.drop(columns=["PerDShap"]).copy()
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv", index=False)

    print(f"[Trial {i}] Saved 3 CSVs:")
    print(f"  - {DATASET_NAME}_{class_tag}_test_{i}.csv")
    print(f"  - {DATASET_NAME}_{class_tag}_{i}.csv")
    print(f"  - {DATASET_NAME}_{class_tag}_TMC_{i}.csv")


' 20) RT-IoT2022 (UCI via ucimlrepo) '
from ucimlrepo import fetch_ucirepo
DATASET_NAME = "rtiot2022"

n_trials = 3
seed0 = 260326

# random K-way task
k_min, k_max = 2, 5

# train size = m, test size = m
m_min, m_max = 900, 1000

# max resampling attempts per trial
max_subset_attempts = 500

# save location
save_dir = r"E:/   Datasets/Datasets_III"   # change if needed
os.makedirs(save_dir, exist_ok=True)

# DShap temp directory
temp_dir = "./temp"
os.makedirs(temp_dir, exist_ok=True)


# =========================================================
# 2) Helper functions
# =========================================================
def tag_from_classes(cls_list):
    return "_".join(map(str, sorted(cls_list)))


def flatten_target(y_obj):
    """
    Convert ucimlrepo targets to a clean 1D pandas Series.
    """
    if isinstance(y_obj, pd.DataFrame):
        if y_obj.shape[1] == 1:
            y = y_obj.iloc[:, 0].copy()
        else:
            # if multiple columns unexpectedly exist, use the first one
            y = y_obj.iloc[:, 0].copy()
    elif isinstance(y_obj, pd.Series):
        y = y_obj.copy()
    else:
        y = pd.Series(y_obj)

    y = y.reset_index(drop=True)
    y.name = "target"
    return y


def drop_id_columns(X):
    """
    Drop only pure record-ID columns, not meaningful network fields like
    id.orig_p or id.resp_p.
    """
    cols_to_drop = []
    for col in X.columns:
        c = str(col).strip().lower()
        if c in ["id", "unnamed: 0", "index"]:
            cols_to_drop.append(col)

    if cols_to_drop:
        print("Dropping ID-like columns:", cols_to_drop)
        X = X.drop(columns=cols_to_drop)

    return X


def choose_feasible_subset(y_pool, classes_pool, rng, k_min, k_max, m_min, m_max, max_attempts=500):
    """
    Randomly regenerate (K, selected_classes, m) until feasible.

    Feasible means:
    - selected classes all have enough examples
    - total required count is 2*m
    - class-wise allocation does not exceed class counts
    """
    class_counts = y_pool.value_counts().to_dict()

    for attempt in range(1, max_attempts + 1):
        K = int(rng.integers(k_min, k_max + 1))
        selected_classes = sorted(rng.choice(classes_pool, size=K, replace=False).tolist())
        m = int(rng.integers(m_min, m_max + 1))

        total_needed = 2 * m
        base = total_needed // K
        rem = total_needed % K
        per_class_counts = [base + (1 if j < rem else 0) for j in range(K)]

        feasible = True
        for j, c in enumerate(selected_classes):
            if class_counts.get(c, 0) < per_class_counts[j]:
                feasible = False
                break

        if feasible:
            return K, selected_classes, m, per_class_counts, attempt

    raise ValueError(
        f"Failed to find a feasible class subset after {max_attempts} attempts. "
        f"Try reducing m_min/m_max or restricting to larger classes."
    )


def preprocess_mixed_tabular(X_train_raw, X_test_raw, random_state):
    """
    Mixed-type preprocessing:
    - numeric: median impute
    - categorical: most_frequent impute + one-hot
    - scale -> PCA -> scale
    """
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train_raw.columns if c not in num_cols]

    # --------------------------
    # numeric
    # --------------------------
    if len(num_cols) > 0:
        X_train_num = X_train_raw[num_cols].copy()
        X_test_num = X_test_raw[num_cols].copy()

        num_imputer = SimpleImputer(strategy="median")
        X_train_num = pd.DataFrame(
            num_imputer.fit_transform(X_train_num),
            columns=num_cols,
            index=X_train_raw.index
        )
        X_test_num = pd.DataFrame(
            num_imputer.transform(X_test_num),
            columns=num_cols,
            index=X_test_raw.index
        )
    else:
        X_train_num = pd.DataFrame(index=X_train_raw.index)
        X_test_num = pd.DataFrame(index=X_test_raw.index)

    # --------------------------
    # categorical
    # --------------------------
    if len(cat_cols) > 0:
        X_train_cat = X_train_raw[cat_cols].copy()
        X_test_cat = X_test_raw[cat_cols].copy()

        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = pd.DataFrame(
            cat_imputer.fit_transform(X_train_cat),
            columns=cat_cols,
            index=X_train_raw.index
        )
        X_test_cat = pd.DataFrame(
            cat_imputer.transform(X_test_cat),
            columns=cat_cols,
            index=X_test_raw.index
        )

        X_train_cat = pd.get_dummies(X_train_cat, columns=cat_cols, drop_first=False)
        X_test_cat = pd.get_dummies(X_test_cat, columns=cat_cols, drop_first=False)

        # align columns
        X_train_cat, X_test_cat = X_train_cat.align(
            X_test_cat, join="left", axis=1, fill_value=0
        )
    else:
        X_train_cat = pd.DataFrame(index=X_train_raw.index)
        X_test_cat = pd.DataFrame(index=X_test_raw.index)

    # --------------------------
    # combine
    # --------------------------
    X_train_proc = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_proc = pd.concat([X_test_num, X_test_cat], axis=1)

    X_train_proc = X_train_proc.astype(float)
    X_test_proc = X_test_proc.astype(float)

    # --------------------------
    # scale -> PCA -> scale
    # --------------------------
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_proc.values)
    X_test_scaled = scaler1.transform(X_test_proc.values)

    pca_dim = min(
        np.random.randint(5, 101),
        X_train_scaled.shape[0],
        X_train_scaled.shape[1]
    )

    pca = PCA(n_components=pca_dim, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    scaler2 = StandardScaler()
    X_train_pca = scaler2.fit_transform(X_train_pca)
    X_test_pca = scaler2.transform(X_test_pca)

    X_train = pd.DataFrame(
        X_train_pca,
        columns=[f"PC{k+1}" for k in range(X_train_pca.shape[1])]
    )
    X_test = pd.DataFrame(
        X_test_pca,
        columns=[f"PC{k+1}" for k in range(X_test_pca.shape[1])]
    )

    return X_train, X_test


# =========================================================
# 3) Load dataset
# =========================================================
print(f"Loading dataset: {DATASET_NAME}")
rt_iot2022 = fetch_ucirepo(id=942)

X_pool = pd.DataFrame(rt_iot2022.data.features).copy()
y_pool_raw = flatten_target(rt_iot2022.data.targets)

print("Original X shape:", X_pool.shape)
print("Original y shape:", y_pool_raw.shape)

# remove ID-like column
X_pool = drop_id_columns(X_pool)

# remove rows with missing label if any
mask = ~y_pool_raw.isna()
X_pool = X_pool.loc[mask].reset_index(drop=True)
y_pool_raw = y_pool_raw.loc[mask].reset_index(drop=True)

# encode labels into integers
y_pool = pd.Series(pd.Categorical(y_pool_raw).codes, name="target")

print("Processed X shape:", X_pool.shape)
print("Class counts:")
print(y_pool.value_counts().sort_index())

classes_pool = sorted(y_pool.unique().tolist())

if len(classes_pool) < k_min:
    raise ValueError(
        f"Dataset has only {len(classes_pool)} classes, but k_min={k_min}."
    )


# =========================================================
# 4) Main trials
# =========================================================
for i in range(1, n_trials + 1):
    print("\n" + "=" * 70)
    print(f"Trial {i}/{n_trials}")
    print("=" * 70)

    rs = seed0 + i
    rng = np.random.default_rng(rs)

    # -----------------------------------------------------
    # choose feasible subset by regeneration
    # -----------------------------------------------------
    K, selected_classes, m, per_class_counts, used_attempt = choose_feasible_subset(
        y_pool=y_pool,
        classes_pool=classes_pool,
        rng=rng,
        k_min=k_min,
        k_max=k_max,
        m_min=m_min,
        m_max=m_max,
        max_attempts=max_subset_attempts
    )

    class_tag = tag_from_classes(selected_classes)

    print(f"Feasible subset found after {used_attempt} attempt(s)")
    print(f"K = {K}")
    print(f"selected_classes = {selected_classes}")
    print(f"m = {m}")
    print(f"per_class_counts = {per_class_counts}")
    print(f"class_tag = {class_tag}")

    # -----------------------------------------------------
    # sample rows from each selected class
    # -----------------------------------------------------
    idx_all = []
    for j, c in enumerate(selected_classes):
        need_c = per_class_counts[j]
        grp = y_pool[y_pool == c]

        idx_c = grp.sample(
            n=need_c,
            random_state=rs + 1007 + int(c)
        ).index

        idx_all.append(idx_c)

    idx = idx_all[0]
    for j in range(1, len(idx_all)):
        idx = idx.union(idx_all[j])

    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all_raw = y_pool.loc[idx].reset_index(drop=True)

    # remap labels to {0, ..., K-1}
    label_map = {c: j for j, c in enumerate(selected_classes)}
    y_all = y_all_raw.map(label_map).astype(int)

    # -----------------------------------------------------
    # stratified split: m train / m test
    # -----------------------------------------------------
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=m,
        test_size=m,
        random_state=rs
    )
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True)

    print("Train class counts:")
    print(y_train.value_counts().sort_index())
    print("Test class counts:")
    print(y_test.value_counts().sort_index())

    # -----------------------------------------------------
    # preprocessing
    # -----------------------------------------------------
    X_train, X_test = preprocess_mixed_tabular(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        random_state=rs
    )

    print("X_train shape after preprocessing:", X_train.shape)
    print("X_test shape after preprocessing:", X_test.shape)

    # -----------------------------------------------------
    # save TEST csv
    # -----------------------------------------------------
    df_te = X_test.copy()
    df_te["target"] = y_test.values
    test_path = f"{save_dir}/{DATASET_NAME}_{class_tag}_test_{i}.csv"
    df_te.to_csv(test_path, index=False)

    # -----------------------------------------------------
    # Per_DShap
    # -----------------------------------------------------
    phi = Per_DShap(
        X_train,
        y_train,
        X_test,
        y_test,
        n_permutations=5000,
        model_builder=lambda: LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            multi_class="auto"
        ),
        random_state=0,
        progress=True
    )

    df_tr = X_train.copy()
    df_tr["target"] = y_train.values
    df_tr["PerDShap"] = phi

    per_path = f"{save_dir}/{DATASET_NAME}_{class_tag}_{i}.csv"
    df_tr.to_csv(per_path, index=False)

    # -----------------------------------------------------
    # TMC Shapley via DShap
    # -----------------------------------------------------
    shapley_model = DShap(
        X_train.values,
        y_train.values,
        X_test.values,
        y_test.values,
        num_test=len(y_test),
        sources=None,
        sample_weight=None,
        model_family="logistic",
        metric="accuracy",
        overwrite=True,
        directory=temp_dir,
        seed=250625 + i,
        max_iter=100,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = X_train.copy()
    df_tmc["target"] = y_train.values
    df_tmc["TMC_Shap"] = shapley_model.vals_tmc

    tmc_path = f"{save_dir}/{DATASET_NAME}_{class_tag}_TMC_{i}.csv"
    df_tmc.to_csv(tmc_path, index=False)

    print(f"[Trial {i}] Saved:")
    print(" ", test_path)
    print(" ", per_path)
    print(" ", tmc_path)

print("\nAll trials finished.")