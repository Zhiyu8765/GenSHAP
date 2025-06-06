'''
0. Working Directory and Packages
'''
from ast import increment_lineno
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal

'Check Working Directory'
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#sys.path.append('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')

'Import Packages'
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



'''
Data shapley values of MNIST (0&1) (Computed by DataShapley) 
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
train_0 = df_train[df_train['label'] == 0].sample(n=500, random_state = 505).index
train_1 = df_train[df_train['label'] == 1].sample(n=500, random_state = 505).index
test_0 = df_test[df_test['label'] == 0].sample(n=500, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=500, random_state = 505).index
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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist01 = './temp_Mnist01'
shapley_Mnist01 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist01, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist01.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist01 = shapley_Mnist01.vals_tmc
G_Shap_Mnist01 = shapley_Mnist01.vals_g
LOO_Mnist01 = shapley_Mnist01.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))

### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist01_High(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist01,
          'G Shapley': G_Shap_Mnist01,
          'LOO':         LOO_Mnist01},
    X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(42, 53), order='descending', random_iters=10, save_path=path)

'Save results'
train01_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train01.csv"
test01_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test01.csv"

train01_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train01_df["label"] = y_train
train01_df["TMC_Shap"] = TMC_Shap_Mnist01
train01_df["G_Shap"] = G_Shap_Mnist01
train01_df["LOO_Shap"] = LOO_Mnist01
train01_df.to_csv(train01_path, index=False)

test01_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test01_df["label"] = y_test
test01_df.to_csv(test01_path, index=False)




'''
Data shapley values of MNIST (1&2) (Computed by DataShapley) 
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
train_1 = df_train[df_train['label'] == 1].sample(n=500, random_state = 505).index
train_2 = df_train[df_train['label'] == 2].sample(n=500, random_state = 505).index
test_1 = df_test[df_test['label'] == 1].sample(n=500, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_1], X_train[train_2]])
y_train = np.concatenate([y_train[train_1], y_train[train_2]])
X_test = np.concatenate([X_test[test_1], X_test[test_2]])
y_test = np.concatenate([y_test[test_1], y_test[test_2]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 1 relabel as 0; 2 relabel as 1'
y_train = np.where(y_train == 1, 0, 1)
y_test = np.where(y_test == 1, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist12 = './temp_Mnist12'
shapley_Mnist12 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist12, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist12.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist12 = shapley_Mnist12.vals_tmc
G_Shap_Mnist12 = shapley_Mnist12.vals_g
LOO_Mnist12 = shapley_Mnist12.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist12_High(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist12,
          'G Shapley': G_Shap_Mnist12,
          'LOO':         LOO_Mnist12},
    X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(40, 50), order='descending', random_iters=10, save_path=path)

'Save results'
train12_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train12.csv"
test12_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test12.csv"

train12_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train12_df["label"] = y_train
train12_df["TMC_Shap"] = TMC_Shap_Mnist12
train12_df["G_Shap"] = G_Shap_Mnist12
train12_df["LOO_Shap"] = LOO_Mnist12
train12_df.to_csv(train12_path, index=False)

test12_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test12_df["label"] = y_test
test12_df.to_csv(test12_path, index=False)




'''
Data shapley values of MNIST (2&3) (Computed by DataShapley) 
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
train_2 = df_train[df_train['label'] == 2].sample(n=500, random_state = 505).index
train_3 = df_train[df_train['label'] == 3].sample(n=500, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=500, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_2], X_train[train_3]])
y_train = np.concatenate([y_train[train_2], y_train[train_3]])
X_test = np.concatenate([X_test[test_2], X_test[test_3]])
y_test = np.concatenate([y_test[test_2], y_test[test_3]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 1 relabel as 0; 2 relabel as 1'
y_train = np.where(y_train == 2, 0, 1)
y_test = np.where(y_test == 2, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist23 = './temp_Mnist23'
shapley_Mnist23 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist23, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist23.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist23 = shapley_Mnist23.vals_tmc
G_Shap_Mnist23 = shapley_Mnist23.vals_g
LOO_Mnist23 = shapley_Mnist23.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist23_Low(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist23,
          'G Shapley': G_Shap_Mnist23,
          'LOO':         LOO_Mnist23}, X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(48, 60), order='ascending', random_iters=10, save_path=path)

'Save results'
train23_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train23.csv"
test23_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test23.csv"

train23_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train23_df["label"] = y_train
train23_df["TMC_Shap"] = TMC_Shap_Mnist23
train23_df["G_Shap"] = G_Shap_Mnist23
train23_df["LOO_Shap"] = LOO_Mnist23
train23_df.to_csv(train23_path, index=False)

test23_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test23_df["label"] = y_test
test23_df.to_csv(test23_path, index=False)



'''
Data shapley values of MNIST (2&3) (Computed by DataShapley) 
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
train_2 = df_train[df_train['label'] == 2].sample(n=500, random_state = 505).index
train_3 = df_train[df_train['label'] == 3].sample(n=500, random_state = 505).index
test_2 = df_test[df_test['label'] == 2].sample(n=500, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_2], X_train[train_3]])
y_train = np.concatenate([y_train[train_2], y_train[train_3]])
X_test = np.concatenate([X_test[test_2], X_test[test_3]])
y_test = np.concatenate([y_test[test_2], y_test[test_3]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 1 relabel as 0; 2 relabel as 1'
y_train = np.where(y_train == 2, 0, 1)
y_test = np.where(y_test == 2, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist23 = './temp_Mnist23'
shapley_Mnist23 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist23, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist23.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist23 = shapley_Mnist23.vals_tmc
G_Shap_Mnist23 = shapley_Mnist23.vals_g
LOO_Mnist23 = shapley_Mnist23.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist23_Low(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist23,
          'G Shapley': G_Shap_Mnist23,
          'LOO':         LOO_Mnist23}, X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(48, 60), order='ascending', random_iters=10, save_path=path)

'Save results'
train23_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train23.csv"
test23_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test23.csv"

train23_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train23_df["label"] = y_train
train23_df["TMC_Shap"] = TMC_Shap_Mnist23
train23_df["G_Shap"] = G_Shap_Mnist23
train23_df["LOO_Shap"] = LOO_Mnist23
train23_df.to_csv(train23_path, index=False)

test23_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test23_df["label"] = y_test
test23_df.to_csv(test23_path, index=False)


'''
Data shapley values of MNIST (3&4) (Computed by DataShapley) 
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
train_3 = df_train[df_train['label'] == 3].sample(n=500, random_state = 505).index
train_4 = df_train[df_train['label'] == 4].sample(n=500, random_state = 505).index
test_3 = df_test[df_test['label'] == 3].sample(n=500, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_3], X_train[train_4]])
y_train = np.concatenate([y_train[train_3], y_train[train_4]])
X_test = np.concatenate([X_test[test_3], X_test[test_4]])
y_test = np.concatenate([y_test[test_3], y_test[test_4]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 3 relabel as 0; 4 relabel as 1'
y_train = np.where(y_train == 3, 0, 1)
y_test = np.where(y_test == 3, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist34 = './temp_Mnist34'
shapley_Mnist34 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist34, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist34.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist34 = shapley_Mnist34.vals_tmc
G_Shap_Mnist34 = shapley_Mnist34.vals_g
LOO_Mnist34 = shapley_Mnist34.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist34_High(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist34,
          'G Shapley': G_Shap_Mnist34,
          'LOO':         LOO_Mnist34}, X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(42, 52), order='descending', random_iters=10, save_path=path)

'Save results'
train34_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train34.csv"
test34_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test34.csv"

train34_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train34_df["label"] = y_train
train34_df["TMC_Shap"] = TMC_Shap_Mnist34
train34_df["G_Shap"] = G_Shap_Mnist34
train34_df["LOO_Shap"] = LOO_Mnist34
train34_df.to_csv(train34_path, index=False)

test34_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test34_df["label"] = y_test
test34_df.to_csv(test34_path, index=False)


'''
Data shapley values of MNIST (4&5) (Computed by DataShapley) 
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
train_4 = df_train[df_train['label'] == 4].sample(n=500, random_state = 505).index
train_5 = df_train[df_train['label'] == 5].sample(n=500, random_state = 505).index
test_4 = df_test[df_test['label'] == 4].sample(n=500, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_4], X_train[train_5]])
y_train = np.concatenate([y_train[train_4], y_train[train_5]])
X_test = np.concatenate([X_test[test_4], X_test[test_5]])
y_test = np.concatenate([y_test[test_4], y_test[test_5]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 3 relabel as 0; 4 relabel as 1'
y_train = np.where(y_train == 4, 0, 1)
y_test = np.where(y_test == 4, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist45 = './temp_Mnist45'
shapley_Mnist45 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist45, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist45.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist45= shapley_Mnist45.vals_tmc
G_Shap_Mnist45 = shapley_Mnist45.vals_g
LOO_Mnist45 = shapley_Mnist45.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist45_Low(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist45,
          'G Shapley': G_Shap_Mnist45,
          'LOO':         LOO_Mnist45}, X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(48, 61), order='ascending', random_iters=10, save_path=path)

'Save results'
train45_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train45.csv"
test45_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test45.csv"

train45_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train45_df["label"] = y_train
train45_df["TMC_Shap"] = TMC_Shap_Mnist45
train45_df["G_Shap"] = G_Shap_Mnist45
train45_df["LOO_Shap"] = LOO_Mnist45
train45_df.to_csv(train45_path, index=False)

test45_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test45_df["label"] = y_test
test45_df.to_csv(test45_path, index=False)


'''
Data shapley values of MNIST (5&6) (Computed by DataShapley) 
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
train_5 = df_train[df_train['label'] == 5].sample(n=500, random_state = 505).index
train_6 = df_train[df_train['label'] == 6].sample(n=500, random_state = 505).index
test_5 = df_test[df_test['label'] == 5].sample(n=500, random_state = 505).index
test_6 = df_test[df_test['label'] == 6].sample(n=500, random_state = 505).index
X_train = np.concatenate([X_train[train_5], X_train[train_6]])
y_train = np.concatenate([y_train[train_5], y_train[train_6]])
X_test = np.concatenate([X_test[test_5], X_test[test_6]])
y_test = np.concatenate([y_test[test_5], y_test[test_6]])
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

'Relabel y: 3 relabel as 0; 4 relabel as 1'
y_train = np.where(y_train == 5, 0, 1)
y_test = np.where(y_test == 5, 0, 1)

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
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory_Mnist56 = './temp_Mnist56'
shapley_Mnist56 = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_Mnist56, seed=250505,
                   max_iter=500, tol=1e-20)
shapley_Mnist56.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_Mnist56= shapley_Mnist56.vals_tmc
G_Shap_Mnist56 = shapley_Mnist56.vals_g
LOO_Mnist56 = shapley_Mnist56.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_Mnist56_High(1000).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_Mnist56,
          'G Shapley': G_Shap_Mnist56,
          'LOO':         LOO_Mnist56}, X_train=X_train, y_train=y_train, X_test=X_test,   y_test=y_test,
    a=0.05, b=0.4, ylim=(42, 51), order='descending', random_iters=10, save_path=path)

'Save results'
train45_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST/train45.csv"
test45_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST//test45.csv"

train45_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train45_df["label"] = y_train
train45_df["TMC_Shap"] = TMC_Shap_Mnist45
train45_df["G_Shap"] = G_Shap_Mnist45
train45_df["LOO_Shap"] = LOO_Mnist45
train45_df.to_csv(train45_path, index=False)

test45_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test45_df["label"] = y_test
test45_df.to_csv(test45_path, index=False)

