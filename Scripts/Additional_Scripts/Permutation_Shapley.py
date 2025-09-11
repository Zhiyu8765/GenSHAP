''' 0. Import Packages '''
import math
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#os.chdir('/tmp/pycharm_project_236/DataShapley')
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

#%matplotlib inline
MEM_DIR = '../'



''' 1. Function to Calculate Permutation-Based Data Shapley Values '''
import numpy as np
import math
from itertools import combinations
from sklearn.linear_model import LogisticRegression

def Per_DShap(X, y, X_test, y_test, n_permutations = 1000,
    model_builder=lambda: LogisticRegression(max_iter=200, n_jobs=1), random_state=None, progress=True):

    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    Xv = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    yv = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    N = len(X_np)
    if len(y_np) != N:
        raise ValueError("X and y must have the same number of rows")

    # Record baseline accuracy
    # classes, counts = np.unique(yv, return_counts=True)
    # majority = classes[np.argmax(counts)]
    # baseline = np.mean(yv == majority)
    baseline = 0.0

    # Utility function V(S)
    def utility(subset_idx):
        if len(subset_idx) == 0:
            return baseline
        ys = y_np[subset_idx]
        if len(np.unique(ys)) < 2:
            return baseline
        clf = model_builder()
        clf.fit(X_np[subset_idx], ys)
        return float(clf.score(Xv, yv))

    rng = np.random.default_rng(random_state)
    phi = np.zeros(N, dtype=float)
    counts = np.zeros(N, dtype=int)

    iterator = range(n_permutations)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Permutation Shapley", unit="perm")
        except Exception:
            pass

    for _ in iterator:
        perm = rng.permutation(N)
        prev_v = baseline
        for j in range(1, N + 1):
            i = perm[j - 1]
            S = perm[:j]
            vj = utility(S)
            mc = vj - prev_v  # marginal contribution of i
            counts[i] += 1
            phi[i] += (mc - phi[i]) / counts[i]
            prev_v = vj

    return phi



''' 2. Calculation of Permutation-Based Shapley Values '''
' a. Iris '
n_trials = 1
# generate m as random number < 50
# m = random.randint(20, 45)
m = 40
a = 0
b = 2
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
num_test = m
directory = './temp'
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/Group03'
iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, name = 'target')
iris_df = pd.concat([X, y], axis=1)
print(iris_df['target'].value_counts())

for i in range(1, n_trials + 1):
    class_a = iris_df[iris_df["target"] == a].sample(n=m, random_state=i)
    class_b = iris_df[iris_df["target"] == b].sample(n=m, random_state=i)
    class_ab = pd.concat([class_a, class_b], axis=0).reset_index(drop=True)
    class_ab['target'] = class_ab['target'].map({a:0, b:1})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        class_ab.drop(columns = ['target']),
        class_ab['target'],
        test_size = 0.5,
        stratify = class_ab['target'],
        random_state = i
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns = X_train_raw.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns = X_test_raw.columns)

    phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
                            model_builder=lambda: LogisticRegression(max_iter=100, n_jobs=1),
                            random_state=0, progress=True)
    Iris_train = X_train.copy()
    Iris_train['target'] = y_train.values
    Iris_train['PerDShap'] = phi
    Iris_train.to_csv(f'{save_dir}/Iris_{a}{b}_{20+i}', index=False)

    Iris_test = X_test.copy()
    Iris_test['target'] = y_test.values
    Iris_test.to_csv(f'{save_dir}/Iris_{a}{b}_test_{20+i}', index=False)

    shapley_model = DShap(X_train.values, y_train.values,
                          X_test.values, y_test.values,
                          num_test, sources=None,
                          sample_weight=None, model_family=model,
                          metric=metric, overwrite=True,
                          directory=directory, seed=250625 + i,
                          max_iter=100, tol=1e-5)
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
    Iris_TMC = Iris_train.copy()
    Iris_TMC['TMC_Shap'] = shapley_model.vals_tmc
    Iris_TMC.to_csv(f'{save_dir}/Iris_{a}{b}_TMC_{20+i}', index=False)

    print(f'[Trial {i}] CSV files saved.')

save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/Per_Trainntest/Group03"
)
plot_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/Per_Trainntest/Group03/Plots"
)
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, n_trials + 1):
    train_f = os.path.join(save_dir, f"Iris_{a}{b}_{20+i}")
    test_f  = os.path.join(save_dir, f"Iris_{a}{b}_test_{20+i}")
    tmc_f = os.path.join(save_dir, f"Iris_{a}{b}_TMC_{20+i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)
    df_tmc = pd.read_csv(tmc_f)

    X_train = df_train.drop(columns=["target", "PerDShap"], errors="ignore").values
    y_train = df_train["target"].values
    per_shap = df_train["PerDShap"].values
    tmc_shap = df_tmc["TMC_Shap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_Iris_{a}{b}_high_{20+i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": per_shap,
            "TMC Shapley":  tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(50, 102),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_Iris_{a}{b}_low_{20+i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": per_shap,
            "TMC Shapley":  tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(50, 102),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")


' b. Breast Cancer '
n_trials = 50
m = 50
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/Group08'
os.makedirs(save_dir, exist_ok=True)

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
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_train_pca)
    X_test = scaler2.transform(X_test_pca)
    X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])

    D_shap = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
                            model_builder=lambda: LogisticRegression(max_iter=100, n_jobs=1),
                            random_state=0, progress=True)

    out_tr = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    out_tr['target'] = y_train.values
    out_tr['PerDShap'] = D_shap
    out_tr.to_csv(f'{save_dir}/bc_{i}', index=False)
    out_te = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
    out_te['target'] = y_test.values
    out_te.to_csv(f'{save_dir}/bc_test_{i}', index=False)

    print(f"[Trial {i}] CSV files saved.")

n_trials = 50
save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/Per_TrainnTest/Group08"
)
plot_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/Per_TrainnTest/Group08"
)
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, n_trials + 1):
    train_f = os.path.join(save_dir, f"bc_{i}")
    test_f  = os.path.join(save_dir, f"bc_test_{i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)

    X_train = df_train.drop(columns=["target", "PerDShap"], errors="ignore").values
    y_train = df_train["target"].values
    true_shap = df_train["PerDShap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_bc_high_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(40, 105),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_bc_low_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(30, 105),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")


' c. MNIST '
n_trials = 20
m        = 50
seed0    = 1
class_a, class_b = 1, 7

save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/Per_Trainntest/Group19"
)
plot_dir = os.path.join(save_dir, "PerformancePlots")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


import tensorflow as tf
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

train_mask = (y_train_full == class_a) | (y_train_full == class_b)
# test_mask  = (y_test_full  == class_a) | (y_test_full  == class_b)

X_pool = X_train_full[train_mask]
y_pool = y_train_full[train_mask]
X_pool = X_pool.reshape(len(X_pool), -1)
X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(y_pool, name='target')
# y_pool = y_pool.reshape(len(y_pool), -1)
# X_test_all  = X_test_full[test_mask]
# # y_test_all  = y_test_full[test_mask]
#
# flatten and normalize
# X_pool = X_pool.reshape(len(X_pool), -1)
# X_test_all  = X_pool.reshape(len(X_pool),  -1)
# scaler    = StandardScaler()
# X_train_all = scaler.fit_transform(X_train_all)
# X_test_all  = scaler.transform(X_test_all)
#
# # PCA to 10 dimensions
# pca = PCA(n_components=5)
# X_train_all = scaler.fit_transform(pca.fit_transform(X_train_all))
# X_test_all  = scaler.transform(pca.transform(X_test_all))
#
# scaler2 = StandardScaler()
# X_train_all = scaler2.fit_transform(X_train_all)
# X_test_all  = scaler2.transform(X_test_all)
#
# # to pandas
# X_train_all = pd.DataFrame(X_train_all)
# y_train_all = pd.Series(y_train_all, name='target')
# X_test_all  = pd.DataFrame(X_test_all)
# y_test_all  = pd.Series(y_test_all,  name='target')


for i in range(1, n_trials+1):
    rs = seed0 + i
    idx_a = y_pool[y_pool == class_a] \
        .sample(n=m, random_state=rs).index
    idx_b = y_pool[y_pool == class_b] \
        .sample(n=m, random_state=rs).index
    idx = idx_a.union(idx_b)
    X_all = X_pool.loc[idx].reset_index(drop=True)
    y_all = y_pool.loc[idx].reset_index(drop=True).map({class_a: 0, class_b: 1})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, stratify=y_all, random_state=rs)
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train_raw)
    X_test_scaled = scaler1.transform(X_test_raw)
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_train_pca)
    X_test = scaler2.transform(X_test_pca)
    X_train = pd.DataFrame(X_train, columns=[f'PC{i + 1}' for i in range(X_train.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f'PC{i + 1}' for i in range(X_test.shape[1])])

    D_Shap = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
                            model_builder=lambda: LogisticRegression(max_iter=100, n_jobs=1),
                            random_state=0, progress=True)

    df_tr = X_train.copy()
    df_tr['target']    = y_train.values
    df_tr['PerDShap'] = D_Shap
    df_tr.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_{i}", index=False)

    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_test_{i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=m, sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250907, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_TMC_{i}", index=False)

    print(f"[Trial {i}] CSVs for classes {class_a}/{class_b} saved.")

n_trials = 11
for i in range(1, n_trials+1):
    name = f"MNIST_{class_a}{class_b}_"
    df_tr = pd.read_csv(f"{save_dir}/{name}{i}")
    df_te = pd.read_csv(f"{save_dir}/{name}test_{i}")
    # df_tmc= pd.read_csv(f"{save_dir}/{name}TMC_{i}")

    X_tr = df_tr.drop(columns=['target','PerDShap'])
    y_tr = df_tr['target']
    true_shap = df_tr['PerDShap'].values
    # tmc_shap  = df_tmc['TMC_Shap'].values

    X_te = df_te.drop(columns=['target'])
    y_te = df_te['target']

    # descending removal
    PerformancePlot_Removal(
        vals={
            'True Data Shapley': true_shap
            # 'TMC Shapley':       tmc_shap
        },
        X_train=X_tr.values, y_train=y_tr.values,
        X_test=X_te.values, y_test=y_te.values,
        a=0.03, b=0.4, ylim=(20,105), order='descending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_a}{class_b}_high_{i}.png"
    )

    # ascending removal
    PerformancePlot_Removal(
        vals={
            'True Data Shapley': true_shap
            # 'TMC Shapley':       tmc_shap
        },
        X_train=X_tr.values, y_train=y_tr.values,
        X_test=X_te.values, y_test=y_te.values,
        a=0.03, b=0.4, ylim=(20,105), order='ascending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_a}{class_b}_low_{i}.png"
    )

    print(f"[Trial {i}] Plots for classes {class_a}/{class_b} saved.")

