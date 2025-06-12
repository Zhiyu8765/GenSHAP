''' 0. Import Packages '''
import pandas as pd
import numpy as np
import math
import itertools
from itertools import combinations
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

from ast import increment_lineno
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal

'Check Working Directory'
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#sys.path.append('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')

'Package and Environments for TMC_Shapley'
%load_ext autoreload
%autoreload 2
import importlib
#import shap_utils
#importlib.reload(shap_utils)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from Shapley import ShapNN
from DShap import DShap
from shap_utils import *
from ucimlrepo import fetch_ucirepo
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#%matplotlib inline
MEM_DIR = './'

''' 1. Function to Calculate Data Shapley Values '''
### Note: Two versions below yield the same results.
# 'Brute Force Version'
# def baseline_accuracy(y_test):
#     return max(np.mean(y_test == c) for c in np.unique(y_test))
#
# def DataShapley(X, y, X_test, y_test):
#     N = len(X)
#     DShap = np.zeros(N)
#     baseline = baseline_accuracy(y_test)
#     for i in tqdm(range(N), desc = "Computing Exact Data Shapley"):
#         marginal_sum = 0
#         others = list(set(range(N)) - {i}) # Create index list
#         for s in range(2, N):
#             weight = 1 / (N * math.comb(N - 1, s)) if s > 0 else 1/N
#             for subset_indices in combinations(others, s):
#                 subset = list(subset_indices)
#
#                 X_s = X[subset]
#                 y_s = y[subset]
#
#                 if len(np.unique(y_s)) < 2:
#                     U_s = baseline
#                 else:
#                     model_s = LogisticRegression(max_iter = 1000).fit(X_s, y_s)
#                     U_s = model_s.score(X_test, y_test)
#
#                 X_si = np.vstack([X_s, X[i]])
#                 y_si = np.append(y_s, y[i])
#
#                 if len(np.unique(y_si)) < 2:
#                     U_si = baseline
#                 else:
#                     model_si = LogisticRegression(max_iter = 1000).fit(X_si, y_si)
#                     U_si = model_si.score(X_test, y_test)
#
#                 marginal_sum += weight * (U_si - U_s)
#         DShap[i] = marginal_sum
#     return DShap

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
            model = LogisticRegression(max_iter=500).fit(X_subset, y_subset)
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


''' 2. Experiment on Iris Dataset '''
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
y = pd.Series(iris.target, name='target')
iris_df = pd.concat([X_scaled, y], axis=1)
iris_0 = iris_df[iris_df["target"] == 0].sample(n = 20, random_state = 1)
iris_1 = iris_df[iris_df["target"] == 1].sample(n = 20, random_state = 1)
iris_2 = iris_df[iris_df["target"] == 2].sample(n = 20, random_state = 1)
iris_01 = pd.concat([iris_0, iris_1], axis = 0)
iris_02 = pd.concat([iris_0, iris_2], axis = 0)
iris_12 = pd.concat([iris_1, iris_2], axis = 0)

'0&1'
X_train01, X_test01, y_train01, y_test01 = train_test_split(iris_01.drop(columns = ['target']), iris_01['target'],
                                                            test_size=0.5, stratify=iris_01['target'], random_state=0)
D_shap = DataShapley(X_train01, y_train01, X_test01, y_test01, n_jobs = -1)
D_shap01 = D_shap
model = LogisticRegression(max_iter = 500).fit(X_train01, y_train01)
print(model.score(X_test01, y_test01))
print(pd.Series(y_train01).value_counts())
print(pd.Series(y_test01).value_counts())
Iris_01 = X_train01.copy()
Iris_01['target'] = y_train01.values
Iris_01['TrueDShap'] = D_shap01
Iris_01.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Iris_01', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
increments = np.arange(0, len(X_train01)+1, 10)
rand_age = np.random.rand(len(X_train01))
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_Iris01_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap01},
            X_train=X_train01.values, y_train=y_train01.values, X_test=X_test01.values, y_test=y_test01.values,
            a=0.02, b=0.5, ylim=(80, 105), order='descending', random_iters=10, save_path=path)

'0&2'
X_train02, X_test02, y_train02, y_test02 = train_test_split(iris_02.drop(columns = ['target']), iris_02['target'],
                                                            test_size=0.5, stratify=iris_02['target'], random_state=0)
D_shap02 = DataShapley(X_train02, y_train02, X_test02, y_test02, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train02, y_train02)
print(model.score(X_test02, y_test02))
print(pd.Series(y_train02).value_counts())
print(pd.Series(y_test02).value_counts())
Iris_02 = X_train02.copy()
Iris_02['target'] = y_train02.values
Iris_02['TrueDShap'] = D_shap02
Iris_02.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Iris_02', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
increments = np.arange(0, len(X_train02)+1, 10)
rand_age = np.random.rand(len(X_train02))
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_Iris02_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap02},
            X_train=X_train02.values, y_train=y_train02.values, X_test=X_test02.values, y_test=y_test02.values,
            a=0.03, b=0.5, ylim=(80, 105), order='descending', random_iters=10, save_path=path)

'1&2'
X_train12, X_test12, y_train12, y_test12 = train_test_split(iris_12.drop(columns = ['target']), iris_12['target'],
                                                            test_size=0.5, stratify=iris_12['target'], random_state=0)
D_shap12 = DataShapley(X_train12, y_train12, X_test12, y_test12, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train12, y_train12)
print(model.score(X_test12, y_test12))
print(pd.Series(y_train12).value_counts())
print(pd.Series(y_test12).value_counts())
Iris_12 = X_train12.copy()
Iris_12['target'] = y_train12.values
Iris_12['TrueDShap'] = D_shap12
Iris_12.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Iris_12', index = False)

increments = np.arange(0, len(X_train12)+1, 10)
rand_age = np.random.rand(len(X_train12))
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/Removal_Iris12_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap12},
            X_train=X_train12.values, y_train=y_train12.values, X_test=X_test12.values, y_test=y_test12.values,
            a=0.03, b=0.5, ylim=(45, 100), order='descending', random_iters=10, save_path=path)




''' 2. Experiment on Breast Cancer Dataset '''
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target, name='target')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
pca = PCA(n_components = 5)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns = [f'PC{i+1}' for i in range(5)])
bc_df = pd.concat([X_pca, y], axis=1)
bc_0 = bc_df[bc_df["target"] == 0]
bc_1 = bc_df[bc_df["target"] == 1]
bc_0_sample_1 = bc_0.sample(n = 10, random_state = 1)
bc_1_sample_1 = bc_1.sample(n = 10, random_state = 1)
bc_0_sample_2 = bc_0.sample(n = 10, random_state = 2)
bc_1_sample_2 = bc_1.sample(n = 10, random_state = 2)
bc_0_sample_3 = bc_0.sample(n = 10, random_state = 3)
bc_1_sample_3 = bc_1.sample(n = 10, random_state = 3)
bc_train1 = pd.concat([bc_0_sample_1, bc_1_sample_1], axis = 0)
bc_train2 = pd.concat([bc_0_sample_2, bc_1_sample_2], axis = 0)
bc_train3 = pd.concat([bc_0_sample_3, bc_1_sample_3], axis = 0)

X_train1, X_test1, y_train1, y_test1 = train_test_split(bc_train1.drop(columns = ['target']), bc_train1['target'],
                                                        test_size=0.5, stratify= bc_train1['target'], random_state=1)
D_shap_1 = DataShapley(X_train1, y_train1, X_test1, y_test1, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train1, y_train1)
print(model.score(X_test1, y_test1)) # 0.9
print(pd.Series(y_train1).value_counts())
print(pd.Series(y_test1).value_counts())
bc_1 = X_train1.copy()
bc_1['target'] = y_train1.values
bc_1['TrueDShap'] = D_shap_1
bc_1.to_csv('bc_1', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train1)+1, 10)
rand_age = np.random.rand(len(X_train1))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/
path = 'Removal_bc1_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_1},
            X_train=X_train1.values, y_train=y_train1.values, X_test=X_test1.values, y_test=y_test1.values,
            a=0.01, b=0.5, ylim=(75, 105), order='ascending', random_iters=10, save_path=path)


X_train2, X_test2, y_train2, y_test2 = train_test_split(bc_train2.drop(columns = ['target']), bc_train2['target'],
                                                        test_size=0.5, stratify= bc_train2['target'], random_state=1)
D_shap_2 = DataShapley(X_train2, y_train2, X_test2, y_test2, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train2, y_train2)
print(model.score(X_test2, y_test2)) # 0.9
print(pd.Series(y_train2).value_counts())
print(pd.Series(y_test2).value_counts())
bc_2 = X_train2.copy()
bc_2['target'] = y_train2.values
bc_2['TrueDShap'] = D_shap_2
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
bc_2.to_csv('bc_2', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train2)+1, 10)
rand_age = np.random.rand(len(X_train2))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = 'Removal_bc2_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_2},
            X_train=X_train2.values, y_train=y_train2.values, X_test=X_test2.values, y_test=y_test2.values,
            a=0.03, b=0.5, ylim=(50, 105), order='descending', random_iters=10, save_path=path)

X_train3, X_test3, y_train3, y_test3 = train_test_split(bc_train3.drop(columns = ['target']), bc_train3['target'],
                                                        test_size=0.5, stratify= bc_train3['target'], random_state=1)
D_shap_3 = DataShapley(X_train3, y_train3, X_test3, y_test3, n_jobs = -1)
model = LogisticRegression(max_iter = 500).fit(X_train3, y_train3)
print(model.score(X_test3, y_test3)) # 0.9
print(pd.Series(y_train3).value_counts())
print(pd.Series(y_test3).value_counts())
bc_3 = X_train3.copy()
bc_3['target'] = y_train3.values
bc_3['TrueDShap'] = D_shap_3
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/
bc_3.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/bc_3', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train3)+1, 10)
rand_age = np.random.rand(len(X_train3))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_bc3_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_3},
            X_train=X_train3.values, y_train=y_train3.values, X_test=X_test3.values, y_test=y_test3.values,
            a=0.03, b=0.5, ylim=(70, 105), order='ascending', random_iters=10, save_path=path)


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
train_0 = df_train[df_train['label'] == 0].sample(n=5, random_state = 505).index
train_1 = df_train[df_train['label'] == 1].sample(n=5, random_state = 505).index
test_0 = df_test[df_test['label'] == 0].sample(n=5, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=5, random_state = 505).index
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
MNIST_01.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_01', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_01_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST01},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(20, 80), order='descending', random_iters=10, save_path=path)

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
train_1 = df_train[df_train['label'] == 1].sample(n=5, random_state = 505).index
train_2 = df_train[df_train['label'] == 2].sample(n=5, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=5, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=5, random_state = 505).index
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
MNIST_12.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_12', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_12_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST12},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(40, 80), order='ascending', random_iters=10, save_path=path)


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
train_1 = df_train[df_train['label'] == 1].sample(n=5, random_state = 505).index
train_2 = df_train[df_train['label'] == 2].sample(n=5, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=5, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=5, random_state = 505).index
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
MNIST_12.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_12', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_12_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST12},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(40, 80), order='ascending', random_iters=10, save_path=path)


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
train_2 = df_train[df_train['label'] == 2].sample(n=5, random_state = 505).index
train_3 = df_train[df_train['label'] == 3].sample(n=5, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=5, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=5, random_state = 505).index
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
MNIST_23.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_23', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_23_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST23},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(30, 90), order='descending', random_iters=10, save_path=path)

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
train_3 = df_train[df_train['label'] == 3].sample(n=5, random_state = 505).index
train_4 = df_train[df_train['label'] == 4].sample(n=5, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=5, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=5, random_state = 505).index
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
MNIST_34.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_34', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_34_Low.png'
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
train_4 = df_train[df_train['label'] == 4].sample(n=5, random_state = 505).index
train_5 = df_train[df_train['label'] == 5].sample(n=5, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=5, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=5, random_state = 505).index
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
MNIST_45.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_45', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_45_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST45},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(40, 85), order='descending', random_iters=10, save_path=path)

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
train_5 = df_train[df_train['label'] == 5].sample(n=5, random_state = 505).index
train_6 = df_train[df_train['label'] == 6].sample(n=5, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=5, random_state = 505).index
test_6 = df_test[df_test['label'] == 6].sample(n=5, random_state = 505).index
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
MNIST_56.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_56', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_56_Low.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST56},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(35, 90), order='ascending', random_iters=10, save_path=path)


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
train_6 = df_train[df_train['label'] == 6].sample(n=5, random_state = 505).index
train_7 = df_train[df_train['label'] == 7].sample(n=5, random_state = 505).index
test_6 = df_test[df_test['label'] == 6].sample(n=5, random_state = 505).index
test_7 = df_test[df_test['label'] == 7].sample(n=5, random_state = 505).index
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
MNIST_67.to_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/MNIST_67', index = False)

' Performance plot (removal based on descending True Data Shapley values) '
increments = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))
#/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250611/
path = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612/Removal_MNIST_67_High.png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': D_shap_MNIST67},
            X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values,
            a=0.01, b=0.5, ylim=(20, 50), order='descending', random_iters=10, save_path=path)
