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

# import inspect
# import shap_utils
# print(inspect.getsource(shap_utils))


'''
1. Data shapley values of Iris (Computed by DataShapley)
'''

'Load data'
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

'Implementation of DataShapley'
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
shapley_Iris = DShap(X_train, y_train, X_test, y_test, num_test,sources = None,
                     sample_weight = None, model_family = model, metric = 'accuracy',
                     overwrite = True, directory = directory, seed = 250324)
shapley_Iris.run(save_every = 100, err = 0.1, g_run = True, loo_run = True)
TMC_Shap_Iris = shapley_Iris.vals_tmc
G_Shap_Iris = shapley_Iris.vals_g
LOO_Iris = shapley_Iris.vals_loo

'Construct dataframe by adding data shapley values'
iris_df = pd.DataFrame(X_train, columns = ['sepal length', 'sepal width',
                                           'petal length', 'petal width'])
iris_df['target'] = y_train
iris_df['TMC_Shapley'] = TMC_Shap_Iris
iris_df['G_Shapley'] = G_Shap_Iris
iris_df['LOO_Shapley'] = LOO_Iris

'Evaluation by plots'
### ----- Plot 0. PCA ----- ###
# Standardized X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Normalized Data Shapley Values
min_shapley = iris_df['TMC_Shapley'].min()
max_shapley = iris_df['TMC_Shapley'].max()
iris_df['TMC_Shapley_Normalized'] = (iris_df['TMC_Shapley'] - min_shapley) / (max_shapley - min_shapley)
plt.figure(figsize = (8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = iris_df['target'],
            s = iris_df['TMC_Shapley_Normalized']*1000, cmap = 'viridis',
            alpha = 0.6, edgecolors = "w", linewidth = 0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA plot with TMC Shapley Value Sizes')
plt.colorbar(label = 'Target Class')
plt.tight_layout()
plt.savefig('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250324/PCA_Iris_DShap(0.1).png', dpi=300, bbox_inches='tight')


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250324/Removal_Iris_High(0.1).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_Iris, G_Shap_Iris, LOO_Iris],
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.05, b = 0.4, order = 'descending', random_iters = 10, save_path=path)



'''
2. Data shapley values of Breast Cancer (Computed by DataShapley)
'''
data_bc   = load_breast_cancer()
X_bc      = pd.DataFrame(data_bc.data, columns=data_bc.feature_names)
y_bc      = pd.Series(data_bc.target, name="target")
df        = pd.concat([X_bc, y_bc], axis=1)
df0       = df[df["target"] == 0]
df1_down  = df[df["target"] == 1].sample(n=len(df0), random_state=42)
df_balanced = pd.concat([df0, df1_down]).sample(frac=1, random_state=42)
X_balanced = df_balanced.drop(columns=["target"])
y_balanced = df_balanced["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.5, random_state=0)
scaler   = MinMaxScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform(X_test)
y_train  = y_train.to_numpy(dtype=np.int64)
y_test   = y_test.to_numpy(dtype=np.int64)

' Algorithm = Logistic Regression'
model = 'logistic'
problem = 'classification'
num_test = 212
directory_bc = './temp_bc'
shapley_bc = DShap(X_train, y_train, X_test, y_test, num_test, sources = None,
                   sample_weight = None, model_family = model,metric = 'accuracy',
                   overwrite = True, directory = directory_bc, seed = 250415,
                   max_iter = 100, tol = 1e-15)
shapley_bc.run(save_every=100, err=0.1, g_run=True, loo_run=True)
# error = 0.1, time (2mins)
# error = 0.01, time (5mins)
# error = 0.001, time (3h30mins)
# error = 0.0005, time (>7h)
# error = 0.0001, time (about 5 days)
TMC_Shap_bc = shapley_bc.vals_tmc
G_Shap_bc = shapley_bc.vals_g
LOO_bc = shapley_bc.vals_loo
increments_bc = np.arange(0, len(X_train)+1, 10)
rand_bc = np.random.rand(len(X_train))

### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250422/Removal_BC_Low(0.01_100).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_bc, G_Shap_bc, LOO_bc],
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, save_path=path)

# ' Algorithm = KNN'
# model = 'KNN'
# problem = 'classification'
# num_test = 285
# directory_bc = './temp_bc'
# shapley_bc4 = DShap(X_train, y_train, X_test, y_test, num_test,
#                     model_family = model, n_neighbors = 1,
#                    metric = 'accuracy', overwrite = True,
#                    directory = directory_bc, seed = 250324)
# shapley_bc4.run(save_every=100, err=0.001, g_run=True, loo_run=True)
# # error = 0.1, time(1min)
# # error = 0.01, time (7min)
# # error = 0.001, time (19:02-)
# # error = 0.0005, time (>7h)
# # error = 0.0001, time (about 5 days)
# TMC_Shap_bc4 = shapley_bc4.vals_tmc
# #G_Shap_bc4 = shapley_bc4.vals_g
# LOO_bc4 = shapley_bc4.vals_loo
# increments_bc = np.arange(0, len(X_train)+1, 10)
# rand_bc = np.random.rand(len(X_train))
#
# ### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
# path = '/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250410/Removal_BC_Low_KNN(0.001).png'
# removal_percentages, results = PerformancePlot_Removal(vals_dict={'TMC Shapley': TMC_Shap_bc4, 'LOO': LOO_bc4},
#                                                        X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
#                                                        a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, seed = 42, save_path=path)


'''
3. Data shapley values of Estimation of Obesity Level (Computed by DataShapley)
'''
'Data Acquisition'
Obe_Level = fetch_ucirepo(id=544)
X_Obe = Obe_Level.data.features
y_Obe = Obe_Level.data.targets
class_counts = y_Obe.value_counts()
print(class_counts)
df_Obe = pd.concat([X_Obe, y_Obe], axis=1)

'Data Pre-Processing'
categories_to_keep = ["Obesity_Type_I", "Obesity_Type_II",
                         "Obesity_Type_III", "Normal_Weight", "Insufficient_Weight"]
df_Obe = df_Obe[df_Obe.iloc[:, -1].isin(categories_to_keep)].reset_index(drop=True)
obesity = ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
df_Obe["Obesity"] = df_Obe.iloc[:, -1].apply(
    lambda x:1 if x in obesity else 0
)
df_Obe.drop(df_Obe.columns[-2], axis = 1, inplace = True)
target_col = df_Obe.columns[-1]
df_class0 = df_Obe[df_Obe[target_col] == 0]
df_class1 = df_Obe[df_Obe[target_col] == 1]
df_class1_down = df_class1.sample(n=559, random_state=42)
df_Obe = pd.concat([df_class1_down, df_class0]).sample(frac=1, random_state=42).reset_index(drop=True)
X_Obe = df_Obe.drop("Obesity", axis = 1)
y_Obe = df_Obe["Obesity"]
class_counts = y_Obe.value_counts()
print(class_counts)

# One-Hot encoding
categorical_cols_idx = [0, 4, 5, 8, 9, 11, 14, 15]
categorical_cols = X_Obe.columns[categorical_cols_idx]
X_Obe = pd.get_dummies(X_Obe, columns = categorical_cols)
# Min-Max Normalisation
numerical_cols = [col for col in X_Obe.columns if col not in categorical_cols]
scaler = StandardScaler()
X_Obe[numerical_cols] = scaler.fit_transform(X_Obe[numerical_cols])
X_train, X_test, y_train, y_test = train_test_split(X_Obe, y_Obe, test_size=0.5, random_state=0)
X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()
y_train = y_train.to_numpy().ravel()
y_test  = y_test.to_numpy().ravel()

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 559
directory_obe = './temp_obe'
shapley_obe = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_obe, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_obe.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_obe = shapley_obe.vals_tmc
G_Shap_obe = shapley_obe.vals_g
LOO_obe = shapley_obe.vals_loo
increments_obe = np.arange(0, len(X_train)+1, 10)
rand_obe = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250425/Removal_OBE_Balanced_mean_High(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_obe, G_Shap_obe, LOO_obe],
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'descending', random_iters = 10, save_path=path)


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
X_train, X_test, y_train, y_test = train_test_split(X_balanced_encoded, y_balanced, test_size=0.5, random_state=0)
X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()
y_train = y_train.to_numpy().ravel()
y_test  = y_test.to_numpy().ravel()

'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 536
directory_aids = './temp_aids'
shapley_aids = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS = shapley_aids.vals_tmc
G_Shap_AIDS = shapley_aids.vals_g
LOO_AIDS = shapley_aids.vals_loo
increments_AIDS = np.arange(0, len(X_train)+1, 10)
rand_AIDS = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250425/Removal_AIDS_Low_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_AIDS, G_Shap_AIDS, LOO_AIDS],
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, save_path=path)



'''
5. Data shapley values of Estimation of Thyroid Disease (Computed by DataShapley) -- not suitable
'''
'Data Acquisition'
AGE = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/AGE/age.csv')
age_col = AGE.iloc[:, 2]
# count_ge_40 = (age_col >= 40).sum()
# count_lt_40 = (age_col < 40).sum()
# print("Number of rows with age >= 40:", count_ge_40)
# print("Number of rows with age < 40:", count_lt_40)
y_AGE = (AGE['RIDAGEYR'] >= 40).astype(int)
X_AGE = AGE.iloc[:, 3:10]

'Data Pre-Processing'
categorical_cols = ['RIAGENDR']
X_AGE = pd.get_dummies(X_AGE, columns=categorical_cols)
numerical_cols = [col for col in X_AGE if col not in categorical_cols]
scaler = MinMaxScaler()
X_AGE[numerical_cols] = scaler.fit_transform(X_AGE[numerical_cols])
X_train, X_test, y_train, y_test = train_test_split(X_AGE, y_AGE, test_size=0.5, random_state=0)
X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()
y_train = y_train.to_numpy(dtype=np.int64)
y_test  = y_test.to_numpy(dtype=np.int64)


'Data Shapley'
model = 'logistic'
problem = 'classification'
num_test = 1139
directory_age = './temp_age'
shapley_age = DShap(X_train, y_train, X_test, y_test, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_age, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_age.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_age = shapley_age.vals_tmc
G_Shap_age = shapley_age.vals_g
LOO_age = shapley_age.vals_loo
increments_age = np.arange(0, len(X_train)+1, 10)
rand_age = np.random.rand(len(X_train))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250422/Removal_AGE_Low(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(vals = [TMC_Shap_age, G_Shap_age, LOO_age],
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, save_path=path)
