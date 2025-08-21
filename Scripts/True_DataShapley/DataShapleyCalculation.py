''' 0. Import Packages '''
' For Calculation of Ground-Truth Data Shapley Values '
import math
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal

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


''' 1. Function to Calculate Data Shapley Values '''
## Note: Two versions below yield the same results.
'Brute Force Version'
# def DataShapley(X, y, X_test, y_test):
#     N = len(X)
#     DShap = np.zeros(N)
#     baseline = max(np.mean(y_test == c) for c in np.unique(y_test))
#
#     def U(subset_indices):
#         size = len(subset_indices)
#         if size == 0:
#             return 0.0
#         X_s = X.iloc[subset_indices]
#         y_s = y.iloc[subset_indices]
#         if size == 1 or len(np.unique(y_s)) < 2:
#             return baseline
#         model = LogisticRegression(max_iter = 100).fit(X_s, y_s)
#         return model.score(X_test, y_test)
#     for i in tqdm(range(N), desc = "Computing Exact Data Shapley(Brute Force Version)"):
#         others = [j for j in range(N) if j != i]
#         marginal_sum = 0.0
#         for s in range(0, N):
#             weight = 1 / (N * math.comb(N - 1, s))
#             for S in combinations(others, s):
#                 U_S = U(list(S))
#                 U_Si = U(list(S) + [i])
#                 marginal_sum += weight * (U_Si - U_S)
#             DShap[i] = marginal_sum
#     return DShap

'Optimised by: memoization of subsets'
def DataShapley(X, y, X_test, y_test):
    N = len(X)
    DShap = np.zeros(N)
    baseline = max(np.mean(y_test == c) for c in np.unique(y_test))
    utility_cache = {}

    def compute_utility(key, subset_indices):
        if key in utility_cache:
            return utility_cache[key]
        size = len(subset_indices)
        if size == 0:
            u = 0.0
        else:
            X_s = X.iloc[subset_indices]
            y_s = y.iloc[subset_indices]
            if size == 1 or len(np.unique(y_s)) < 2:
                u = baseline
            else:
                u = LogisticRegression(max_iter = 100).fit(X_s, y_s).score(X_test, y_test)
        utility_cache[key] = u
        return u

    for i in tqdm(range(N), desc="Computing Exact Data Shapley (Memoization)"):
        marginal_sum = 0.0
        others = [j for j in range(N) if j != i]

        for s in range(0, N):
            weight = 1 / (N * math.comb(N - 1, s))
            for S in combinations(others, s):
                key_S = tuple(sorted(S))
                key_Si = tuple(sorted(S + (i,)))
                U_s = compute_utility(key_S, list(S))
                U_si = compute_utility(key_Si, list(S) + [i])
                marginal_sum += weight * (U_si - U_s)
        DShap[i] = marginal_sum

    return DShap

# 'Optimised by: memoization of subsets & parallel processing'
# def DataShapley(X, y, X_test, y_test, n_jobs=-1):
#     N = len(X)
#     DShap = np.zeros(N)
#     baseline = max(np.mean(y_test == c) for c in np.unique(y_test))
#     cache = {}
#
#     def compute_U(key, subset_indices):
#         if key in cache:
#             return cache[key]
#         size = len(subset_indices)
#         if size == 0:
#             u = 0.0
#         else:
#             X_s = X.iloc[subset_indices]; y_s = y.iloc[subset_indices]
#             if size == 1 or len(np.unique(y_s)) < 2:
#                 u = baseline
#             else:
#                 u = LogisticRegression(max_iter=50).fit(X_s, y_s).score(X_test, y_test)
#         cache[key] = u
#         return u
#
#     def process_subset(i, s, S):
#         key_S  = tuple(sorted(S))
#         key_Si = tuple(sorted(S + [i]))
#         U_S  = compute_U(key_S, S)
#         U_Si = compute_U(key_Si, S + [i])
#         weight = 1/(N * math.comb(N-1, s))
#         return weight * (U_Si - U_S)
#
#     for i in tqdm(range(N), desc="Parallel DataShapley"):
#         others = [j for j in range(N) if j != i]
#         tasks = [(i, s, list(S))
#                  for s in range(0, N)
#                  for S in combinations(others, s)]
#         results = Parallel(n_jobs=n_jobs)(delayed(process_subset)(*t) for t in tasks)
#         DShap[i] = sum(results)
#     return DShap
#
#
# 'Optimised by: memoization of subsets & parallel processing & batch processing'
# def DataShapley(X, y, X_test, y_test, n_jobs=-1, batch_size=100):
#     N = len(X)
#     DShap = np.zeros(N)
#     baseline = max(np.mean(y_test == c) for c in np.unique(y_test))
#     cache = {}
#
#     def compute_U(key, subset_indices):
#         if key in cache:
#             return cache[key]
#         size = len(subset_indices)
#         if size == 0:
#             u = 0.0
#         else:
#             X_s = X.iloc[subset_indices]; y_s = y.iloc[subset_indices]
#             if size == 1 or len(np.unique(y_s)) < 2:
#                 u = baseline
#             else:
#                 u = LogisticRegression(max_iter=50).fit(X_s, y_s).score(X_test, y_test)
#         cache[key] = u
#         return u
#
#     def batch_process(i, tasks):
#         total = 0.0
#         for s, S in tasks:
#             key_S  = tuple(sorted(S))
#             key_Si = tuple(sorted(S + [i]))
#             U_S  = compute_U(key_S, S)
#             U_Si = compute_U(key_Si, S + [i])
#             weight = 1/(N * math.comb(N-1, s))
#             total += weight * (U_Si - U_S)
#         return total
#
#     for i in tqdm(range(N), desc="Batch DataShapley"):
#         others = [j for j in range(N) if j != i]
#         tasks = []
#         for s in range(0, N):
#             for S in combinations(others, s):
#                 tasks.append((s, list(S)))
#         # partition into batches
#         batches = [tasks[k:k+batch_size] for k in range(0, len(tasks), batch_size)]
#         results = Parallel(n_jobs=n_jobs)(delayed(batch_process)(i, b) for b in batches)
#         DShap[i] = sum(results)
#     return DShap


''' 2. Experiment on Iris Dataset '''
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
# y = pd.Series(iris.target, name='target')
# iris_df = pd.concat([X_scaled, y], axis=1)
# iris_0 = iris_df[iris_df["target"] == 0].sample(n = 15, random_state = 1)
# iris_1 = iris_df[iris_df["target"] == 1].sample(n = 15, random_state = 1)
# iris_2 = iris_df[iris_df["target"] == 2].sample(n = 15, random_state = 1)
# iris_01 = pd.concat([iris_0, iris_1], axis = 0)
# iris_02 = pd.concat([iris_0, iris_2], axis = 0)
# iris_12 = pd.concat([iris_1, iris_2], axis = 0)
#
# '0&1'
# X_train01, X_test01, y_train01, y_test01 = train_test_split(iris_01.drop(columns = ['target']), iris_01['target'],
#                                                             test_size=0.5, stratify=iris_01['target'], random_state=0)
# D_shap = DataShapley(X_train01, y_train01, X_test01, y_test01, n_jobs = -1)
# D_shap2 = DataShapley2(X_train01, y_train01, X_test01, y_test01, n_jobs = -1, batch_size = 100)
# D_shap01 = D_shap
# model = LogisticRegression(max_iter = 500).fit(X_train01, y_train01)
# print(model.score(X_test01, y_test01))
# print(pd.Series(y_train01).value_counts())
# print(pd.Series(y_test01).value_counts())
# Iris_01 = X_train01.copy()
# Iris_01['target'] = y_train01.values
# Iris_01['TrueDShap'] = D_shap01
# #Iris_01.to_csv('/tmp/pycharm_project_236/Outputs/250618/Trial06/Iris_01_10pts_CPU.csv', index = False)
# Iris_01.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_01', index = False)
# Iris_01_test = X_test01.copy()
# Iris_01_test['target'] = y_test01.values
# Iris_01_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_01_test', index = False)
# model = 'logistic'
# problem = 'classification'
# num_test = 15
# directory = './temp'
# shapley_Iris01 = DShap(X_train01.values, y_train01.values, X_test01.values, y_test01.values, num_test, sources = None,
#                    sample_weight = None, model_family = model,metric = 'accuracy',
#                    overwrite = True, directory = directory, seed = 250612,
#                    max_iter = 100, tol = 1e-5)
# shapley_Iris01.run(save_every = 100, err = 0.1, g_run = True, loo_run = True)
# Iris_01_TMC = Iris_01.copy()
# Iris_01_TMC['TMC_Shap'] = shapley_Iris01.vals_tmc
# ##Export dataset
# Iris_01_TMC.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_01_TMC', index = False)
#
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
# increments = np.arange(0, len(X_train01)+1, 10)
# rand_age = np.random.rand(len(X_train01))
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_Iris01_Low.png'
# #path = '/tmp/pycharm_project_236/Outputs/250618/Trial06/Removal_Iris01_10pts_Low_CPU.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap01},
#             X_train=X_train01.values, y_train=y_train01.values, X_test=X_test01.values, y_test=y_test01.values,
#             a=0.02, b=0.5, ylim=(20, 105), order='ascending', random_iters=10, save_path=path)
#
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
# increments = np.arange(0, len(X_train01)+1, 10)
# rand_age = np.random.rand(len(X_train01))
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_Iris01_High_TMC.png'
# #path = '/tmp/pycharm_project_236/Outputs/250618/Trial06/Removal_Iris01_10pts_Low_CPU.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap01,
#           'TMC Shapley': shapley_Iris01.vals_tmc},
#             X_train=X_train01.values, y_train=y_train01.values, X_test=X_test01.values, y_test=y_test01.values,
#             a=0.02, b=0.5, ylim=(20, 105), order='descending', random_iters=10, save_path=path)
#
# '0&2'
# X_train02, X_test02, y_train02, y_test02 = train_test_split(iris_02.drop(columns = ['target']), iris_02['target'],
#                                                             test_size=0.5, stratify=iris_02['target'], random_state=0)
# D_shap02 = DataShapley(X_train02, y_train02, X_test02, y_test02, n_jobs = -1)
# model = LogisticRegression(max_iter = 500).fit(X_train02, y_train02)
# print(model.score(X_test02, y_test02))
# print(pd.Series(y_train02).value_counts())
# print(pd.Series(y_test02).value_counts())
# Iris_02 = X_train02.copy()
# Iris_02['target'] = y_train02.values
# Iris_02['TrueDShap'] = D_shap02
# Iris_02.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_02', index = False)
# Iris_02_test = X_test02.copy()
# Iris_02_test['target'] = y_test02.values
# Iris_02_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_02_test', index = False)
#
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
# increments = np.arange(0, len(X_train02)+1, 10)
# rand_age = np.random.rand(len(X_train02))
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_Iris02_Low.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap02},
#             X_train=X_train02.values, y_train=y_train02.values, X_test=X_test02.values, y_test=y_test02.values,
#             a=0.03, b=0.5, ylim=(20, 105), order='ascending', random_iters=10, save_path=path)
#
# '1&2'
# X_train12, X_test12, y_train12, y_test12 = train_test_split(iris_12.drop(columns = ['target']), iris_12['target'],
#                                                             test_size=0.5, stratify=iris_12['target'], random_state=0)
# D_shap12 = DataShapley(X_train12, y_train12, X_test12, y_test12, n_jobs = -1)
# model = LogisticRegression(max_iter = 500).fit(X_train12, y_train12)
# print(model.score(X_test12, y_test12)) # 0.9
# print(pd.Series(y_train12).value_counts())
# print(pd.Series(y_test12).value_counts())
# Iris_12 = X_train12.copy()
# Iris_12['target'] = y_train12.values
# Iris_12['TrueDShap'] = D_shap12
# Iris_12.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12', index = False)
# Iris_12_test = X_test12.copy()
# Iris_12_test['target'] = y_test12.values
# Iris_12_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_test', index = False)
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
# increments = np.arange(0, len(X_train12)+1, 10)
# rand_age = np.random.rand(len(X_train12))
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_Iris12_Low.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap12},
#             X_train=X_train12.values, y_train=y_train12.values, X_test=X_test12.values, y_test=y_test12.values,
#             a=0.03, b=0.5, ylim=(50, 105), order='ascending', random_iters=10, save_path=path)
#
# ##Calculation of TMC Shapley values
# y_train12 = y_train12.replace({1: 0, 2: 1})
# y_test12 = y_test12.replace({1: 0, 2: 1})
# model = 'logistic'
# problem = 'classification'
# num_test = 15
# directory = './temp'
# shapley_Iris12 = DShap(X_train12.values, y_train12.values, X_test12.values, y_test12.values, num_test, sources = None,
#                    sample_weight = None, model_family = model,metric = 'accuracy',
#                    overwrite = True, directory = directory, seed = 250612,
#                    max_iter = 100, tol = 1e-5)
# shapley_Iris12.run(save_every = 100, err = 0.1, g_run = True, loo_run = True)
# Iris_12_TMC = Iris_12.copy()
# Iris_12_TMC['TMC_Shap'] = shapley_Iris12.vals_tmc
# ##Export dataset
# Iris_12_TMC.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_TMC', index = False)
#
# increments = np.arange(0, len(X_train12)+1, 10)
# rand_age = np.random.rand(len(X_train12))
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_Iris12_Low_TMC.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap12,
#           'TMC Data Shapley': shapley_Iris12.vals_tmc},
#             X_train=X_train12.values, y_train=y_train12.values, X_test=X_test12.values, y_test=y_test12.values,
#             a=0.03, b=0.5, ylim=(60, 100), order='ascending', random_iters=10, save_path=path)

'Automatic Generation'
n_trials = 20
m = 10
a = 1
b = 2
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group28'
#save_dir = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V3'
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
num_test = m
directory = './temp'

iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, name = 'target')
iris_df = pd.concat([X, y], axis=1)

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

    D_Shap = DataShapley(X_train, y_train, X_test, y_test)
    Iris_train = X_train.copy()
    Iris_train['target'] = y_train.values
    Iris_train['TrueDShap'] = D_Shap
    Iris_train.to_csv(f'{save_dir}/Iris_{a}{b}_{i}', index=False)

    Iris_test = X_test.copy()
    Iris_test['target'] = y_test.values
    Iris_test.to_csv(f'{save_dir}/Iris_{a}{b}_test_{i}', index=False)

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
    Iris_TMC.to_csv(f'{save_dir}/Iris_{a}{b}_TMC_{i}', index=False)
    print(f'[Trial {i}] CSV files saved.')

n_trials = 20
# a = 1
# b = 2
save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group28"
)
plot_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group28"
)
os.makedirs(plot_dir, exist_ok=True)
for i in range(1, n_trials + 1):
    train_f = os.path.join(save_dir, f"Iris_{a}{b}_{i}")
    test_f  = os.path.join(save_dir, f"Iris_{a}{b}_test_{i}")
    tmc_f   = os.path.join(save_dir, f"Iris_{a}{b}_TMC_{i}")

    df_train = pd.read_csv(train_f)
    df_test  = pd.read_csv(test_f)
    df_tmc   = pd.read_csv(tmc_f)

    X_train = df_train.drop(columns=["target", "TrueDShap"], errors="ignore").values
    y_train = df_train["target"].values
    true_shap = df_train["TrueDShap"].values
    tmc_shap = df_tmc["TMC_Shap"].values

    X_test = df_test.drop(columns=["target"], errors="ignore").values
    y_test = df_test["target"].values

    out_png_high = os.path.join(plot_dir, f"Removal_Iris_{a}{b}_high_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(20, 105),order="descending",
        random_iters=10, save_path=out_png_high,
    )
    out_png_low = os.path.join(plot_dir, f"Removal_Iris_{a}{b}_low_{i}.png")
    removal_percents, results = PerformancePlot_Removal(
        vals={
            "True Data Shapley": true_shap,
            "TMC Shapley":       tmc_shap
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, a=0.03, b=0.4, ylim=(50, 105),order="ascending",
        random_iters=10, save_path=out_png_low,
    )

    print(f"[Trial {i}] Plot saved.")


''' 2. Experiment on Breast Cancer Dataset '''
# bc = load_breast_cancer()
# X = pd.DataFrame(bc.data, columns=bc.feature_names)
# y = pd.Series(bc.target, name='target')
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
# pca = PCA(n_components = 5)
# X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns = [f'PC{i+1}' for i in range(5)])
# bc_df = pd.concat([X_pca, y], axis=1)
# bc_0 = bc_df[bc_df["target"] == 0]
# bc_1 = bc_df[bc_df["target"] == 1]
# bc_0_sample_1 = bc_0.sample(n = 15, random_state = 1)
# bc_1_sample_1 = bc_1.sample(n = 15, random_state = 1)
# bc_0_sample_2 = bc_0.sample(n = 15, random_state = 2)
# bc_1_sample_2 = bc_1.sample(n = 15, random_state = 2)
# bc_0_sample_3 = bc_0.sample(n = 15, random_state = 3)
# bc_1_sample_3 = bc_1.sample(n = 15, random_state = 3)
# bc_train1 = pd.concat([bc_0_sample_1, bc_1_sample_1], axis = 0)
# bc_train2 = pd.concat([bc_0_sample_2, bc_1_sample_2], axis = 0)
# bc_train3 = pd.concat([bc_0_sample_3, bc_1_sample_3], axis = 0)
#
# X_train1, X_test1, y_train1, y_test1 = train_test_split(bc_train1.drop(columns = ['target']), bc_train1['target'],
#                                                         test_size=0.5, stratify= bc_train1['target'], random_state=1)
# D_shap_1 = DataShapley(X_train1, y_train1, X_test1, y_test1, n_jobs = -1)
# model = LogisticRegression(max_iter = 500).fit(X_train1, y_train1)
# print(model.score(X_test1, y_test1)) # 0.9
# print(pd.Series(y_train1).value_counts())
# print(pd.Series(y_test1).value_counts())
# bc_1 = X_train1.copy()
# bc_1['target'] = y_train1.values
# bc_1['TrueDShap'] = D_shap_1
# bc_1.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_1', index = False)
# bc_1_test = X_test1.copy()
# bc_1_test['target'] = y_test1.values
# bc_1_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_1_test', index = False)
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# increments = np.arange(0, len(X_train1)+1, 10)
# rand_age = np.random.rand(len(X_train1))
# #/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/bc_1_Removal_High'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap_1},
#             X_train=X_train1.values, y_train=y_train1.values, X_test=X_test1.values, y_test=y_test1.values,
#             a=0.01, b=0.5, ylim=(25, 105), order='descending', random_iters=10, save_path=path)
#
#
# X_train2, X_test2, y_train2, y_test2 = train_test_split(bc_train2.drop(columns = ['target']), bc_train2['target'],
#                                                         test_size=0.5, stratify= bc_train2['target'], random_state=1)
# D_shap_2 = DataShapley(X_train2, y_train2, X_test2, y_test2, n_jobs = -1)
# model = LogisticRegression(max_iter = 500).fit(X_train2, y_train2)
# print(model.score(X_test2, y_test2)) # 0.9
# print(pd.Series(y_train2).value_counts())
# print(pd.Series(y_test2).value_counts())
# bc_2 = X_train2.copy()
# bc_2['target'] = y_train2.values
# bc_2['TrueDShap'] = D_shap_2
# #/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
# bc_2.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_2', index = False)
# bc_2_test = X_test2.copy()
# bc_2_test['target'] = y_test2.values
# bc_2_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_2_test', index = False)
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# increments = np.arange(0, len(X_train2)+1, 10)
# rand_age = np.random.rand(len(X_train2))
# #/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_bc2_Low.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap_2},
#             X_train=X_train2.values, y_train=y_train2.values, X_test=X_test2.values, y_test=y_test2.values,
#             a=0.03, b=0.5, ylim=(50, 105), order='ascending', random_iters=10, save_path=path)
#
# X_train3, X_test3, y_train3, y_test3 = train_test_split(bc_train3.drop(columns = ['target']), bc_train3['target'],
#                                                         test_size=0.5, stratify= bc_train3['target'], random_state=1)
# D_shap_3 = DataShapley(X_train3, y_train3, X_test3, y_test3, n_jobs = -1)
# model = LogisticRegression(max_iter = 500).fit(X_train3, y_train3)
# print(model.score(X_test3, y_test3)) # 0.9
# print(pd.Series(y_train3).value_counts())
# print(pd.Series(y_test3).value_counts())
# bc_3 = X_train3.copy()
# bc_3['target'] = y_train3.values
# bc_3['TrueDShap'] = D_shap_3
# #/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
# bc_3.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_3', index = False)
# bc_3_test = X_test3.copy()
# bc_3_test['target'] = y_test3.values
# bc_3_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/bc_3_test', index = False)
#
# ' Performance plot (removal based on descending True Data Shapley values) '
# increments = np.arange(0, len(X_train3)+1, 10)
# rand_age = np.random.rand(len(X_train3))
# #/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
# path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_bc3_High.png'
# removal_percentages, results = PerformancePlot_Removal(
#     vals={'True Data Shapley': D_shap_3},
#             X_train=X_train3.values, y_train=y_train3.values, X_test=X_test3.values, y_test=y_test3.values,
#             a=0.03, b=0.5, ylim=(50, 105), order='descending', random_iters=10, save_path=path)
#

n_trials = 50
m = 10
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group29'
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
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_train_pca)
    X_test = scaler2.transform(X_test_pca)
    X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])

    D_shap = DataShapley(X_train, y_train, X_test, y_test)

    out_tr = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
    out_tr['target'] = y_train.values
    out_tr['TrueDShap'] = D_shap
    out_tr.to_csv(f'{save_dir}/bc_{i}', index=False)
    out_te = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
    out_te['target'] = y_test.values
    out_te.to_csv(f'{save_dir}/bc_test_{i}', index=False)

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

n_trials = 50
save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group29"
)
plot_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group29"
)
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



'''
True Data shapley values of MNIST 
'''
n_trials = 20
m        = 10
seed0    = 1
class_a, class_b = 9, 0

save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group39"
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

    D_Shap = DataShapley(X_train, y_train, X_test, y_test)

    df_tr = X_train.copy()
    df_tr['target']    = y_train.values
    df_tr['TrueDShap'] = D_Shap
    df_tr.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_{i}", index=False)

    df_te = X_test.copy()
    df_te['target'] = y_test.values
    df_te.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_test_{i}", index=False)

    shapley_model = DShap(
        X_train.values, y_train.values, X_test.values, y_test.values,
        num_test=m, sources=None, sample_weight=None,
        model_family='logistic', metric='accuracy', overwrite=True,
        directory='./temp', seed=250625+i, max_iter=100, err=0.005, tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    df_tmc = df_tr.copy()
    df_tmc['TMC_Shap'] = shapley_model.vals_tmc
    df_tmc.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_TMC_{i}", index=False)

    print(f"[Trial {i}] CSVs for classes {class_a}/{class_b} saved.")

n_trials = 20
for i in range(1, n_trials+1):
    name = f"MNIST_{class_a}{class_b}_"
    df_tr = pd.read_csv(f"{save_dir}/{name}{i}")
    df_te = pd.read_csv(f"{save_dir}/{name}test_{i}")
    df_tmc= pd.read_csv(f"{save_dir}/{name}TMC_{i}")

    X_tr = df_tr.drop(columns=['target','TrueDShap'])
    y_tr = df_tr['target']
    true_shap = df_tr['TrueDShap'].values
    tmc_shap  = df_tmc['TMC_Shap'].values

    X_te = df_te.drop(columns=['target'])
    y_te = df_te['target']

    # descending removal
    PerformancePlot_Removal(
        vals={
            'True Data Shapley': true_shap,
            'TMC Shapley':       tmc_shap
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
            'True Data Shapley': true_shap,
            'TMC Shapley':       tmc_shap
        },
        X_train=X_tr.values, y_train=y_tr.values,
        X_test=X_te.values, y_test=y_te.values,
        a=0.03, b=0.4, ylim=(20,105), order='ascending',
        random_iters=10,
        save_path=f"{plot_dir}/Removal_MNIST_{class_a}{class_b}_low_{i}.png"
    )

    print(f"[Trial {i}] Plots for classes {class_a}/{class_b} saved.")

