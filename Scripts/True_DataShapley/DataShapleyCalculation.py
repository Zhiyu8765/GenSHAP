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

    for i in tqdm(range(N), desc="Computing Exact Data Shapley (Sequential)"):
        marginal_sum = 0.0
        others = [j for j in range(N) if j != i]

        for s in range(2, N):
            weight = 1 / (N * math.comb(N - 1, s))
            for subset in combinations(others, s):
                subset = list(subset)
                key_s = tuple(sorted(subset))
                key_si = tuple(sorted(subset + [i]))

                X_s = X.iloc[subset]
                y_s = y.iloc[subset]
                X_si = pd.concat([X_s, X.iloc[[i]]], axis=0)
                y_si = pd.concat([y_s, y.iloc[[i]]], axis=0)

                U_s = compute_utility(X_s, y_s, key_s)
                U_si = compute_utility(X_si, y_si, key_si)

                marginal_sum += weight * (U_si - U_s)

        DShap[i] = marginal_sum

    return DShap

'Optimised by: memoization of subsets & parallel processing'
def baseline_accuracy(y_test):
    return max(np.mean(y_test == c) for c in np.unique(y_test))

def DataShapley(X, y, X_test, y_test, n_jobs = -1):
    N = len(X)
    DShap = np.zeros(N)
    baseline = baseline_accuracy(y_test)
    utility_cache = {}

    def compute_utility(X_subset, y_subset, key):
        if key in utility_cache:
            return utility_cache[key]
        if len(np.unique(y_subset)) < 2:
            utility = baseline
        else:
            model = LogisticRegression(max_iter=50).fit(X_subset, y_subset)
            utility = model.score(X_test, y_test)
        utility_cache[key] = utility
        return utility

    for i in tqdm(range(N), desc = "Computing Exact Data Shapley"):
        marginal_sum = 0
        others = list(set(range(N)) - {i})

        def process_subset(s, subset_indices):
            subset = list(subset_indices)
            key_s = tuple(sorted(subset))
            key_si = tuple(sorted(subset) + [i])

            X_s = X.iloc[subset]
            y_s = y.iloc[subset]
            X_si = pd.concat([X_s, X.iloc[[i]]], axis=0)
            y_si = pd.concat([y_s, y.iloc[[i]]], axis=0)

            U_s = compute_utility(X_s, y_s, key_s)
            U_si = compute_utility(X_si, y_si, key_si)

            weight = 1/(N * math.comb(N - 1, s))
            return weight * (U_si - U_s)
        results = Parallel(n_jobs = n_jobs)(
            delayed(process_subset)(s, subset_indices)
            for s in range(2, N)
            for subset_indices in combinations(others, s)
        )
        DShap[i] = sum(results)
    return DShap

'Optimised by: memoization of subsets & parallel processing & batch processing'
def baseline_accuracy(y_test):
    return max(np.mean(y_test == c) for c in np.unique(y_test))

def DataShapley(X, y, X_test, y_test, n_jobs = -1, batch_size = 100):
    N = len(X)
    DShap = np.zeros(N)
    baseline = baseline_accuracy(y_test)
    utility_cache = {}

    def compute_utility(X_subset, y_subset, key):
        if key in utility_cache:
            return utility_cache[key]
        if len(np.unique(y_subset)) < 2:
            utility = baseline
        else:
            model = LogisticRegression(max_iter=50).fit(X_subset, y_subset)
            utility = model.score(X_test, y_test)
        utility_cache[key] = utility
        return utility

    def batch_process(i, batch):
        results = []
        for s, subset_indices in batch:
            subset = list(subset_indices)
            key_s = tuple(sorted(subset))
            key_si = tuple(sorted(subset) + [i])

            X_s = X.iloc[subset]
            X_si = pd.concat([X_s, X.iloc[[i]]], axis=0)
            y_s = y.iloc[subset]
            y_si = pd.concat([y_s, y.iloc[[i]]], axis=0)

            U_s = compute_utility(X_s, y_s, key_s)
            U_si = compute_utility(X_si, y_si, key_si)
            weight = 1/(N * math.comb(N - 1, s))
            results.append(weight * (U_si - U_s))
        return sum(results)

    for i in tqdm(range(N), desc = "Computing Exact Data Shapley"):
        others = list(set(range(N)) - {i})
        all_tasks = []
        for s in range(2, N):
            combs = list(combinations(others, s))
            all_tasks.extend([(s, subset) for subset in combs])
        batches = [all_tasks[j:j + batch_size] for j in range(0, len(all_tasks), batch_size)]

        batch_results = Parallel(n_jobs = n_jobs)(
            delayed(batch_process)(i, batch) for batch in batches
        )
        DShap[i] = sum(batch_results)

    return DShap


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
n_trials = 1
m = 15
save_dir = '/Datasets/TrainnTestSets_V3/Group04'
#save_dir = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V3'
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
num_test = m
directory = './temp'

iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, name = 'target')
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
iris_df = pd.concat([X_scaled, y], axis=1)

for i in range(1, n_trials + 1):
    iris_1 = iris_df[iris_df["target"] == 1].sample(n = m, random_state = i)
    iris_2 = iris_df[iris_df["target"] == 2].sample(n = m, random_state = i)
    iris_12 = pd.concat([iris_1, iris_2], axis = 0)
    iris_12['target'] = iris_12['target'].map({1:0, 2:1})

    X_train, X_test, y_train, y_test = train_test_split(
        iris_12.drop(columns = ['target']),
        iris_12['target'],
        test_size = 0.5,
        stratify = iris_12['target'],
        random_state = i
    )

    D_Shap = DataShapley(X_train, y_train, X_test, y_test)
    Iris_train = X_train.copy()
    Iris_train['target'] = y_train.values
    Iris_train['TrueDShap'] = D_Shap
    Iris_train.to_csv(f'{save_dir}/Iris_12_{i}', index=False)

    Iris_test = X_test.copy()
    Iris_test['target'] = y_test.values
    Iris_test.to_csv(f'{save_dir}/Iris_12_test_{i}', index=False)

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
    Iris_TMC.to_csv(f'{save_dir}/Iris_12_TMC_{i}', index=False)
    print(f'[Trial {i}] CSV files saved.')



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

n_trials = 3
m = 15
save_dir = '/Datasets/TrainnTestSets_V3/Group05'
directory = './temp'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(directory, exist_ok=True)

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target, name='target')
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
pca = PCA(n_components=5)
X_pca = pd.DataFrame(
    pca.fit_transform(X_scaled),
    columns=[f'PC{i+1}' for i in range(5)]
)
bc_df = pd.concat([X_pca, y], axis=1)
bc_0 = bc_df[bc_df.target == 0]
bc_1 = bc_df[bc_df.target == 1]

for i in range(1, n_trials + 1):
    # sample 15 from each class
    bc0 = bc_0.sample(n=m, random_state=29+i)
    bc1 = bc_1.sample(n=m, random_state=29+i)
    bc_train = pd.concat([bc0, bc1], axis=0).reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        bc_train.drop(columns='target'),
        bc_train.target,
        test_size=0.5,
        stratify=bc_train.target,
        random_state=29+i
    )
    D_shap = DataShapley(X_tr, y_tr, X_te, y_te, n_jobs=-1)

    out_tr = X_tr.copy()
    out_tr['target'] = y_tr.values
    out_tr['TrueDShap'] = D_shap
    out_tr.to_csv(f'{save_dir}/bc_{29+i}.csv', index=False)

    out_te = X_te.copy()
    out_te['target'] = y_te.values
    out_te.to_csv(f'{save_dir}/bc_{29+i}_test.csv', index=False)

    shapley_model = DShap(
        X_tr.values, y_tr.values,
        X_te.values, y_te.values,
        m,
        sources=None,
        sample_weight=None,
        model_family='logistic',
        metric='accuracy',
        overwrite=True,
        directory=directory,
        seed=250625 + i,
        max_iter=100,
        tol=1e-5
    )
    shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

    out_tr_tmc = out_tr.copy()
    out_tr_tmc['TMC_Shap'] = shapley_model.vals_tmc
    out_tr_tmc.to_csv(f'{save_dir}/bc_{29+i}_TMC.csv', index=False)

    print(f"[Trial {29+i}] CSV files saved: bc_{29+i}.csv, bc_{29+i}_test.csv, bc_{29+i}_TMC.csv")



'''
True Data shapley values of MNIST (0&1) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 0) | (y_train == 1)
test_filter = (y_test == 0) | (y_test == 1)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_0 = df_train[df_train['label'] == 0].sample(n=8, random_state = 505).index
train_1 = df_train[df_train['label'] == 1].sample(n=8, random_state = 505).index
test_0 = df_test[df_test['label'] == 0].sample(n=8, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_0], X_train[train_1]])
y_train = np.concatenate([y_train[train_0], y_train[train_1]])
X_test = np.concatenate([X_test[test_0], X_test[test_1]])
y_test = np.concatenate([y_test[test_0], y_test[test_1]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST01 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.5
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_01 = X_train.copy()
MNIST_01['target'] = y_train.values
MNIST_01['TrueDShap'] = D_shap_MNIST01
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_01.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_01', index = False)
MNIST_01_test = X_test.copy()
MNIST_01_test['target'] = y_test
(MNIST_01_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_01_test', index = False))


' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_01_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST01},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(20, 80), order='ascending', random_iters=10, save_path=path)

'''
True Data shapley values of MNIST (1&2) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 1) | (y_train == 2)
test_filter = (y_test == 1) | (y_test == 2)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_1 = df_train[df_train['label'] == 1].sample(n=8, random_state = 505).index
train_2 = df_train[df_train['label'] == 2].sample(n=8, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=8, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_1], X_train[train_2]])
y_train = np.concatenate([y_train[train_1], y_train[train_2]])
X_test = np.concatenate([X_test[test_1], X_test[test_2]])
y_test = np.concatenate([y_test[test_1], y_test[test_2]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST12 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.5
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_12 = X_train.copy()
MNIST_12['target'] = y_train.values
MNIST_12['TrueDShap'] = D_shap_MNIST12
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_12.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_12', index = False)
MNIST_12_test = X_test.copy()
MNIST_12_test['target'] = y_test
(MNIST_12_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_12_test', index = False))

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_12_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST12},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(10, 50), order='descending', random_iters=10, save_path=path)


'''
True Data shapley values of MNIST (2&3) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 2) | (y_train == 3)
test_filter = (y_test == 2) | (y_test == 3)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_2 = df_train[df_train['label'] == 2].sample(n=8, random_state = 505).index
train_3 = df_train[df_train['label'] == 3].sample(n=8, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=8, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_2], X_train[train_3]])
y_train = np.concatenate([y_train[train_2], y_train[train_3]])
X_test = np.concatenate([X_test[test_2], X_test[test_3]])
y_test = np.concatenate([y_test[test_2], y_test[test_3]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST23 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.7
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_23 = X_train.copy()
MNIST_23['target'] = y_train.values
MNIST_23['TrueDShap'] = D_shap_MNIST23
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_23.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_23', index = False)
MNIST_23_test = X_test.copy()
MNIST_23_test['target'] = y_test
MNIST_23_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_23_test', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_23_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST23},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(30, 90), order='ascending', random_iters=10, save_path=path)

'''
True Data shapley values of MNIST (3&4) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 3) | (y_train == 4)
test_filter = (y_test == 3) | (y_test == 4)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_3 = df_train[df_train['label'] == 3].sample(n=8, random_state = 505).index
train_4 = df_train[df_train['label'] == 4].sample(n=8, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=8, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_3], X_train[train_4]])
y_train = np.concatenate([y_train[train_3], y_train[train_4]])
X_test = np.concatenate([X_test[test_3], X_test[test_4]])
y_test = np.concatenate([y_test[test_3], y_test[test_4]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST34 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.6
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_34 = X_train.copy()
MNIST_34['target'] = y_train.values
MNIST_34['TrueDShap'] = D_shap_MNIST34
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_34.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_34', index = False)
MNIST_34_test = X_test.copy()
MNIST_34_test['target'] = y_test
MNIST_34_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_34_test', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_34_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST34},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(40, 70), order='ascending', random_iters=10, save_path=path)

'''
True Data shapley values of MNIST (4&5) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 4) | (y_train == 5)
test_filter = (y_test == 4) | (y_test == 5)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_4 = df_train[df_train['label'] == 4].sample(n=8, random_state = 505).index
train_5 = df_train[df_train['label'] == 5].sample(n=8, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=8, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_4], X_train[train_5]])
y_train = np.concatenate([y_train[train_4], y_train[train_5]])
X_test = np.concatenate([X_test[test_4], X_test[test_5]])
y_test = np.concatenate([y_test[test_4], y_test[test_5]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST45 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.7
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_45 = X_train.copy()
MNIST_45['target'] = y_train.values
MNIST_45['TrueDShap'] = D_shap_MNIST45
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_45.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_45', index = False)
MNIST_45_test = X_test.copy()
MNIST_45_test['target'] = y_test
MNIST_45_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_45_test', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_45_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST45},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(40, 85), order='ascending', random_iters=10, save_path=path)

'''
True Data shapley values of MNIST (5&6) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 5) | (y_train == 6)
test_filter = (y_test == 5) | (y_test == 6)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_5 = df_train[df_train['label'] == 5].sample(n=8, random_state = 505).index
train_6 = df_train[df_train['label'] == 6].sample(n=8, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=8, random_state = 505).index
test_6 = df_test[df_test['label'] == 6].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_5], X_train[train_6]])
y_train = np.concatenate([y_train[train_5], y_train[train_6]])
X_test = np.concatenate([X_test[test_5], X_test[test_6]])
y_test = np.concatenate([y_test[test_5], y_test[test_6]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST56 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.4
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_56 = X_train.copy()
MNIST_56['target'] = y_train.values
MNIST_56['TrueDShap'] = D_shap_MNIST56
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_56.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_56', index = False)
MNIST_56_test = X_test.copy()
MNIST_56_test['target'] = y_test
MNIST_56_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_56_test', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_56_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST56},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(20, 70), order='ascending', random_iters=10, save_path=path)


'''
True Data shapley values of MNIST (6&7) 
'''
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
train_filter = (y_train == 6) | (y_train == 7)
test_filter = (y_test == 6) | (y_test == 7)
x_train = X_train[train_filter]
y_train = y_train[train_filter]
x_test = X_test[test_filter]
y_test = y_test[test_filter]
df_train = pd.DataFrame({'label': y_train})
df_test = pd.DataFrame({'label': y_test})
train_6 = df_train[df_train['label'] == 6].sample(n=8, random_state = 505).index
train_7 = df_train[df_train['label'] == 7].sample(n=8, random_state = 505).index
test_6 = df_test[df_test['label'] == 6].sample(n=8, random_state = 505).index
test_7 = df_test[df_test['label'] == 7].sample(n=8, random_state = 505).index
X_train = np.concatenate([X_train[train_6], X_train[train_7]])
y_train = np.concatenate([y_train[train_6], y_train[train_7]])
X_test = np.concatenate([X_test[test_6], X_test[test_7]])
y_test = np.concatenate([y_test[test_6], y_test[test_7]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Flatten the images'
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

'Standardisation'
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

'PCA'
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

D_shap_MNIST67 = DataShapley(X_train, y_train, X_test, y_test, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train, y_train)
print(model.score(X_test, y_test)) # 0.4
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
MNIST_67 = X_train.copy()
MNIST_67['target'] = y_train.values
MNIST_67['TrueDShap'] = D_shap_MNIST67
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
MNIST_67.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_67', index = False)
MNIST_67_test = X_test.copy()
MNIST_67_test['target'] = y_test
MNIST_67_test.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/MNIST_67_test', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250623/PerformancePlots/Removal_MNIST_67_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST67},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(20, 70), order='ascending', random_iters=10, save_path=path)
