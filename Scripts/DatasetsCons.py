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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#%matplotlib inline
MEM_DIR = './'

#import inspect
#import shap_utils
#print(inspect.getsource(shap_utils))


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
data_bc = load_breast_cancer()
X_bc = data_bc.data
y_bc = data_bc.target
X_train, X_test, y_train, y_test = train_test_split(X_bc, y_bc, test_size=0.5, random_state=0)

' Algorithm = Logistic Regression'
model = 'logistic'
problem = 'classification'
num_test = 285
directory_bc = './temp_bc'
shapley_bc = DShap(X_train, y_train, X_test, y_test, num_test, sources = None,
                   sample_weight = None, model_family = model,metric = 'accuracy',
                   overwrite = True, directory = directory_bc, seed = 250415,
                   max_iter = 100, tol = 1e-12)
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
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250410/Removal_BC_Low(0.1).png'
removal_percentages, results = PerformancePlot_Removal(vals_dict={'TMC Shapley': TMC_Shap_bc, 'LOO': LOO_bc},
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, seed = 42, save_path=path)

' Algorithm = KNN'
model = 'KNN'
problem = 'classification'
num_test = 285
directory_bc = './temp_bc'
shapley_bc4 = DShap(X_train, y_train, X_test, y_test, num_test,
                    model_family = model, n_neighbors = 1,
                   metric = 'accuracy', overwrite = True,
                   directory = directory_bc, seed = 250324)
shapley_bc4.run(save_every=100, err=0.001, g_run=True, loo_run=True)
# error = 0.1, time(1min)
# error = 0.01, time (7min)
# error = 0.001, time (19:02-)
# error = 0.0005, time (>7h)
# error = 0.0001, time (about 5 days)
TMC_Shap_bc4 = shapley_bc4.vals_tmc
#G_Shap_bc4 = shapley_bc4.vals_g
LOO_bc4 = shapley_bc4.vals_loo
increments_bc = np.arange(0, len(X_train)+1, 10)
rand_bc = np.random.rand(len(X_train))

### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250410/Removal_BC_Low_KNN(0.001).png'
removal_percentages, results = PerformancePlot_Removal(vals_dict={'TMC Shapley': TMC_Shap_bc4, 'LOO': LOO_bc4},
                                                       X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                       a = 0.03, b = 0.4, order = 'ascending', random_iters = 10, seed = 42, save_path=path)
