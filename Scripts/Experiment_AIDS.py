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
4. Data shapley values of Estimation of AIDS (Computed by DataShapley) -- suitable
'''
'Data Acquisition'
AIDS = fetch_ucirepo(id=890)
X_AIDS = AIDS.data.features
y_AIDS = AIDS.data.targets
class_counts = y_AIDS.value_counts()
print(class_counts)
df = pd.concat([X_AIDS, y_AIDS], axis=1)
target_col = y_AIDS.columns[0]
df_class0 = df[df[target_col] == 0]
df_class1 = df[df[target_col] == 1]
df_class0_down = df_class0.sample(n=550, random_state=42)
df_balanced = pd.concat([df_class0_down, df_class1]).sample(frac=1, random_state=42).reset_index(drop=True)
X_balanced = df_balanced.drop(columns=[target_col])
y_balanced = df_balanced[[target_col]]

'Data Pre-Processing'
categorical_cols = ['trt', 'hemo', 'homo', 'drugs', 'oprior', 'z30', 'zprior',
                    'race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt']
X_balanced_encoded = pd.get_dummies(X_balanced, columns=categorical_cols)

numerical_cols = [col for col in X_balanced.columns if col not in categorical_cols]
scaler = StandardScaler()
X_balanced_encoded[numerical_cols] = scaler.fit_transform(X_balanced_encoded[numerical_cols])
X_temp, X_test, y_temp, y_test = train_test_split(
    X_balanced_encoded, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
X_train1, X_temp2, y_train1, y_temp2 = train_test_split(
    X_temp, y_temp, test_size=0.66, random_state=42, stratify=y_temp)
X_train2, X_train3, y_train2, y_train3 = train_test_split(
    X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2)
X_train1 = X_train1.to_numpy()
X_train2 = X_train2.to_numpy()
X_train3 = X_train3.to_numpy()
X_test  = X_test.to_numpy()
y_train1 = y_train1.to_numpy().ravel()
y_train2 = y_train2.to_numpy().ravel()
y_train3 = y_train3.to_numpy().ravel()
y_test  = y_test.to_numpy().ravel()

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 215
directory_aids1 = './temp_aids1'
shapley_aids1 = DShap(X_train1, y_train1, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids1, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids1.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS1 = shapley_aids1.vals_tmc
G_Shap_AIDS1 = shapley_aids1.vals_g
LOO_AIDS1 = shapley_aids1.vals_loo
increments_AIDS1 = np.arange(0, len(X_train1)+1, 10)
rand_AIDS1 = np.random.rand(len(X_train1))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250425/Removal_AIDS1_High_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_AIDS1, G_Shap_AIDS1, LOO_AIDS1],
                                                       X_train = X_train1, y_train = y_train1, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'descending', random_iters = 10, save_path=path)


'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 215
directory_aids2 = './temp_aids2'
shapley_aids2 = DShap(X_train2, y_train2, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids2, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids2.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS2 = shapley_aids2.vals_tmc
G_Shap_AIDS2 = shapley_aids2.vals_g
LOO_AIDS2 = shapley_aids2.vals_loo
increments_AIDS2 = np.arange(0, len(X_train2)+1, 10)
rand_AIDS2 = np.random.rand(len(X_train2))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250507/Removal_AIDS2_Low_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_AIDS2, G_Shap_AIDS2, LOO_AIDS2],
                                                       X_train = X_train2, y_train = y_train2, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, save_path=path)


'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 215
directory_aids3 = './temp_aids3'
shapley_aids3 = DShap(X_train3, y_train3, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids3, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids3.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS3 = shapley_aids3.vals_tmc
G_Shap_AIDS3 = shapley_aids3.vals_g
LOO_AIDS3 = shapley_aids3.vals_loo
increments_AIDS3 = np.arange(0, len(X_train3)+1, 10)
rand_AIDS3 = np.random.rand(len(X_train3))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250507/Removal_AIDS3_High_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_AIDS3, G_Shap_AIDS3, LOO_AIDS3],
                                                       X_train = X_train3, y_train = y_train3, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'descending', random_iters = 10, save_path=path)

'Save results'
train1_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley/train1.csv"
train2_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//train2.csv"
train3_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//train3.csv"
test_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//test.csv"

train1_df = pd.DataFrame(X_train1, columns=[f"feature_{i}" for i in range(X_train1.shape[1])])
train1_df["label"] = y_train1
train1_df["TMC_Shap"] = TMC_Shap_AIDS1
train1_df["G_Shap"] = G_Shap_AIDS1
train1_df["LOO_Shap"] = LOO_AIDS1
train1_df.to_csv(train1_path, index=False)

train2_df = pd.DataFrame(X_train2, columns=[f"feature_{i}" for i in range(X_train2.shape[1])])
train2_df["label"] = y_train2
train2_df["TMC_Shap"] = TMC_Shap_AIDS2
train2_df["G_Shap"] = G_Shap_AIDS2
train2_df["LOO_Shap"] = LOO_AIDS2
train2_df.to_csv(train2_path, index=False)

train3_df = pd.DataFrame(X_train3, columns=[f"feature_{i}" for i in range(X_train3.shape[1])])
train3_df["label"] = y_train3
train3_df["TMC_Shap"] = TMC_Shap_AIDS3
train3_df["G_Shap"] = G_Shap_AIDS3
train3_df["LOO_Shap"] = LOO_AIDS3
train3_df.to_csv(train3_path, index=False)

test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test_df["label"] = y_test
test_df.to_csv(test_path, index=False)



'Extraction of train sets using columns'
random_state = 505
columns_4 = X_balanced_encoded.sample(n=10, axis=1, random_state=random_state).columns
columns_5 = X_balanced_encoded.sample(n=10, axis=1, random_state=random_state + 1).columns
columns_6 = X_balanced_encoded.sample(n=10, axis=1, random_state=random_state + 2).columns
columns_test = X_balanced_encoded.sample(n=10, axis = 1, random_state = random_state + 3).columns
X_train4 = X_balanced_encoded[columns_4].sample(n=500, random_state=random_state).to_numpy()
y_train4 = y_balanced.loc[X_balanced_encoded[columns_4].sample(n=300, random_state=random_state).index].to_numpy().ravel()
X_train5 = X_balanced_encoded[columns_5].sample(n=500, random_state=random_state + 1).to_numpy()
X_test2 = X_balanced_encoded[columns_test].sample(n=500, random_state = random_state + 3).to_numpy()
y_train5 = y_balanced.loc[X_balanced_encoded[columns_5].sample(n=300, random_state=random_state + 1).index].to_numpy().ravel()
X_train6 = X_balanced_encoded[columns_6].sample(n=500, random_state=random_state + 2).to_numpy()
y_train6 = y_balanced.loc[X_balanced_encoded[columns_6].sample(n=300, random_state=random_state + 2).index].to_numpy().ravel()
y_test2 = y_balanced.loc[X_balanced_encoded[columns_6].sample(n=300, random_state = random_state + 4).index].to_numpy().ravel()

'Data Shapley: train4'
model = 'logistic'
problem = 'classification'
num_test = 300
directory_aids4 = './temp_aids4'
shapley_aids4 = DShap(X_train4, y_train4, X_test2, y_test2, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids4, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids4.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS4 = shapley_aids4.vals_tmc
G_Shap_AIDS4 = shapley_aids4.vals_g
LOO_AIDS4 = shapley_aids4.vals_loo
increments_AIDS4 = np.arange(0, len(X_train4)+1, 10)
rand_AIDS4 = np.random.rand(len(X_train4))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250507/Removal_AIDS4_High_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_AIDS4, G_Shap_AIDS4, LOO_AIDS4],
                                                       X_train = X_train4, y_train = y_train4, X_test = X_test2, y_test = y_test2,
                                                       a = 0.03, b = 0.4, order = 'descending', random_iters = 10, save_path=path)