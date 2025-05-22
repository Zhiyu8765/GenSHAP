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
X_train1 = X_train1.to_numpy() # 291*39
X_train2 = X_train2.to_numpy() # 282*39
X_train3 = X_train3.to_numpy() # 283*39
X_test  = X_test.to_numpy() # 215*39
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

# 'Save results'
# train1_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley/train1.csv"
# train2_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//train2.csv"
# train3_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//train3.csv"
# test_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley//test.csv"
#
# train1_df = pd.DataFrame(X_train1, columns=[f"feature_{i}" for i in range(X_train1.shape[1])])
# train1_df["label"] = y_train1
# train1_df["TMC_Shap"] = TMC_Shap_AIDS1
# train1_df["G_Shap"] = G_Shap_AIDS1
# train1_df["LOO_Shap"] = LOO_AIDS1
# train1_df.to_csv(train1_path, index=False)
#
# train2_df = pd.DataFrame(X_train2, columns=[f"feature_{i}" for i in range(X_train2.shape[1])])
# train2_df["label"] = y_train2
# train2_df["TMC_Shap"] = TMC_Shap_AIDS2
# train2_df["G_Shap"] = G_Shap_AIDS2
# train2_df["LOO_Shap"] = LOO_AIDS2
# train2_df.to_csv(train2_path, index=False)
#
# train3_df = pd.DataFrame(X_train3, columns=[f"feature_{i}" for i in range(X_train3.shape[1])])
# train3_df["label"] = y_train3
# train3_df["TMC_Shap"] = TMC_Shap_AIDS3
# train3_df["G_Shap"] = G_Shap_AIDS3
# train3_df["LOO_Shap"] = LOO_AIDS3
# train3_df.to_csv(train3_path, index=False)
#
# test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
# test_df["label"] = y_test
# test_df.to_csv(test_path, index=False)


# '''
# Implementation of GNN
# '''
# import os
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv
#
# DATA_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley'
# GRAPH_DIR = os.path.join(DATA_DIR, 'Graphs_250507')
# os.makedirs(GRAPH_DIR, exist_ok=True)
#
# TRAIN1_CSV = os.path.join(DATA_DIR, 'train1.csv')
# TRAIN2_CSV = os.path.join(DATA_DIR, 'train2.csv')
# TRAIN3_CSV = os.path.join(DATA_DIR, 'train3.csv')
# PRED_CSV_PATH = os.path.join(DATA_DIR, 'train3_predictions.csv')
# TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
#
# def calculate_average_correlation(df: pd.DataFrame) -> float:
#     feats = df.drop(columns=['label', 'G_Shap', 'LOO_Shap', 'TMC_Shap']).select_dtypes(include=[np.number])
#     if feats.shape[1] < 2:
#         return 0.0
#     corr = feats.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     return upper.stack().mean()
#
# def build_nx_graph(df: pd.DataFrame, threshold: float) -> nx.Graph:
#     feats = df.drop(columns=['label', 'G_Shap', 'LOO_Shap']).select_dtypes(include=[np.number])
#     row_corr = feats.T.corr().values
#     G = nx.Graph()
#
#     for idx, row in df.iterrows():
#         node_feats = feats.loc[idx].tolist() + [float(row['TMC_Shap'])]
#         G.add_node(idx, features=node_feats, label=int(row['label']))
#
#     for i in range(len(df)):
#         for j in range(i+1, len(df)):
#             if abs(row_corr[i, j]) >= threshold:
#                 G.add_edge(i, j, weight=row_corr[i, j])
#     return G
#
# def csv_to_pyg_data(path: str, include_target: bool):
#     df = pd.read_csv(path)
#     feat_df = df.drop(columns=['label', 'G_Shap', 'LOO_Shap', 'TMC_Shap']).select_dtypes(include=[np.number])
#     x = torch.tensor(feat_df.values, dtype=torch.float)
#     y = torch.tensor(df['TMC_Shap'].values, dtype=torch.float).unsqueeze(1) if include_target else None
#
#     thr = calculate_average_correlation(df)
#     rc = feat_df.T.corr().values
#     src, dst = [], []
#     for i in range(len(df)):
#         for j in range(i+1, len(df)):
#             if abs(rc[i, j]) >= thr:
#                 src.append(i)
#                 dst.append(j)
#     edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
#     return Data(x=x, edge_index=edge_index, y=y), df
#
# class EnhancedGCN(torch.nn.Module):
#     def __init__(self, in_feats, hidden=512, dropout=0.5):
#         super().__init__()
#         self.conv1 = GCNConv(in_feats, hidden)
#         self.conv2 = GCNConv(hidden, hidden)
#         self.conv3 = GCNConv(hidden, 1)
#         self.dropout = dropout
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return self.conv3(x, edge_index)
# def train_model(model, data1, data2, device, epochs=2000, lr=1e-10):
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = torch.nn.MSELoss()
#     loader = DataLoader([data1, data2], batch_size=2, shuffle=True)
#
#     model.train()
#     for epoch in range(1, epochs + 1):
#         total_loss = 0
#         for batch in loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             out = model(batch)
#             loss = loss_fn(out, batch.y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * batch.num_nodes
#         if epoch % 50 == 0:
#             avg_loss = total_loss / (len(data1.x) + len(data2.x))
#             print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}")
#
# # Save predictions
# def save_predictions(model, data3, df3, device, save_path):
#     model.eval()
#     with torch.no_grad():
#         preds = model(data3.to(device)).cpu().squeeze().numpy()
#     df3['Predicted_TMC_Shap'] = preds
#     df3.to_csv(save_path, index=False)
#     print(f"Predictions saved to {save_path}")
#
# # Train
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data1, _ = csv_to_pyg_data(TRAIN1_CSV, include_target=True)
# data2, _ = csv_to_pyg_data(TRAIN2_CSV, include_target=True)
# data3, df3 = csv_to_pyg_data(TRAIN3_CSV, include_target=False)
# model = EnhancedGCN(in_feats=data1.num_node_features)
# train_model(model, data1, data2, device, epochs=1000, lr=1e-10)
# save_predictions(model, data3, df3, device, PRED_CSV_PATH)
#
# # Performance Plots
# from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
#
# df3_pred = pd.read_csv(PRED_CSV_PATH)
# TMC_true = df3_pred['TMC_Shap'].values
# LOO_vals = df3_pred['LOO_Shap'].values
# G_true = df3_pred['G_Shap'].values
# pred = df3_pred['Predicted_TMC_Shap']
#
# X_train3 = pd.get_dummies(df3_pred.drop(columns=['label', 'G_Shap', 'LOO_Shap', 'TMC_Shap', 'Predicted_TMC_Shap'])).to_numpy()
# y_train3 = df3_pred['label'].to_numpy()
#
# df_test = pd.read_csv(TEST_CSV)
# X_test = pd.get_dummies(df_test.drop(columns=['label'])).to_numpy()
# y_test = df_test['label'].to_numpy()
#
# PERF_PLOT_PNG = os.path.join(DATA_DIR, 'AIDS3pred_Low_2000epoch512hidden.png')
# PerformancePlot_Removal(
#     vals={'TMC Shapley': TMC_true, 'G Shapley': G_true, 'Predicted Shapley': pred, 'LOO': LOO_vals},
#     X_train=X_train3, y_train=y_train3, X_test=X_test, y_test=y_test,
#     a=0.03, b=0.4, ylim=(70, 90), order='ascending', random_iters=10, save_path=PERF_PLOT_PNG
# )


'Experiment on scaling methods of Data Shapley Values'
DATA_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS_DataShapley'
GRAPH_DIR = os.path.join(DATA_DIR, 'Graphs_250507')
os.makedirs(GRAPH_DIR, exist_ok=True)

TRAIN1_CSV = os.path.join(DATA_DIR, 'train1.csv')
TRAIN2_CSV = os.path.join(DATA_DIR, 'train2.csv')
TRAIN3_CSV = os.path.join(DATA_DIR, 'train3.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

Train_1 = pd.read_csv(TRAIN1_CSV)
TMC_train1 = Train_1['TMC_Shap'].values
G_train1 = Train_1['G_Shap'].values
LOO_vals = Train_1['LOO_Shap'].values

X_train1 = pd.get_dummies(Train_1.drop(columns=['label', 'G_Shap', 'LOO_Shap', 'TMC_Shap'])).to_numpy()
y_train1 = Train_1['label'].to_numpy()
df_test = pd.read_csv(TEST_CSV)
X_test = pd.get_dummies(df_test.drop(columns=['label'])).to_numpy()
y_test = df_test['label'].to_numpy()

'Min-Max Normalisation on TMC_Shap'
scaler = MinMaxScaler()
TMC_Scaled1 =scaler.fit_transform(TMC_train1.reshape(1, -1)).flatten()

PERF_PLOT_PNG = os.path.join(DATA_DIR, 'AIDS1_Minmax_High.png')
PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_train1, 'TMC Shapley(scaled by Min-Max)': TMC_Scaled1},
    X_train=X_train1, y_train=y_train1, X_test=X_test, y_test=y_test,
    a=0.03, b=0.4, ylim=(50, 82), order='descending', random_iters=10, save_path=PERF_PLOT_PNG)

'Z-Score '
scaler_z = StandardScaler()
TMC_Zscore = scaler_z.fit_transform(TMC_train1.reshape(1, -1)).flatten()

PERF_PLOT_PNG = os.path.join(DATA_DIR, 'AIDS1_Zscore_Low.png')
PerformancePlot_Removal(
    vals = {'TMC Shapley': TMC_train1, 'TMC Shapley(scaled by Z-Score)': TMC_Zscore},
    X_train = X_train1, y_train = y_train1, X_test = X_test, y_test = y_test,
    a = 0.03, b = 0.4, ylim = (78, 85), order = 'ascending', random_iters = 10, save_path = PERF_PLOT_PNG
)

'Transformation by Percentage Contribution'
phi = sum(abs(TMC_train1))
TMC_Per = TMC_train1/phi

PERF_PLOT_PNG = os.path.join(DATA_DIR, 'AIDS1_Per_High1.png')
PerformancePlot_Removal(
    vals = {'TMC Shapley': TMC_train1, 'TMC Shapley(scaled by Percentage)': TMC_Per},
    X_train = X_train1, y_train = y_train1, X_test = X_test, y_test = y_test,
    a = 0.03, b = 0.4, ylim = (60, 81), order = 'descending', random_iters = 10, save_path = PERF_PLOT_PNG
)
#, 'TMC Shapley(scaled by Percentage)': TMC_Per

'Transformation by Percentage Utility'
full_model = LogisticRegression(max_iter = 500)
full_model.fit(X_train1, y_train1)
U = accuracy_score(y_test, full_model.predict(X_test)) - 0.5
TMC_Utility = TMC_train1/U

PERF_PLOT_PNG = os.path.join(DATA_DIR, 'AIDS1_Utility_Low.png')
PerformancePlot_Removal(
    vals = {'TMC_Shapley': TMC_train1, 'TMC Shapley(scaled by Utility)': TMC_Utility},
    X_train = X_train1, y_train = y_train1, X_test = X_test, y_test = y_test,
    a = 0.03, b = 0.4, ylim = (78, 85), order = 'ascending', random_iters = 10, save_path = PERF_PLOT_PNG
)
#, 'TMC Shapley(scaled by Utility)': TMC_Utility





'Extraction of train sets using columns'
'Read Data'
AIDS = fetch_ucirepo(id=890)
X_AIDS = AIDS.data.features
y_AIDS = AIDS.data.targets
# class_counts = y_AIDS.value_counts()
# print(class_counts)
df = pd.concat([X_AIDS, y_AIDS], axis=1)
target_col = y_AIDS.columns[0]
df_class0 = df[df[target_col] == 0]
df_class1 = df[df[target_col] == 1]
df_class0_down = df_class0.sample(n=520, random_state=42)
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
'Extraction'
random_state = 505
def make_split(X, y, random_seed):
    sampled_idx = X.sample(n=1000, random_state=random_seed).index
    sampled_cols = X.sample(n=20, axis=1, random_state=random_seed).columns
    X_sub = X.loc[sampled_idx, sampled_cols].to_numpy()
    y_sub = y.loc[sampled_idx].to_numpy().ravel()
    return train_test_split(X_sub, y_sub, test_size=0.5, random_state=random_seed, stratify=y_sub)
X_train4, X_test4, y_train4, y_test4 = make_split(X_balanced_encoded, y_balanced, random_state)
X_train5, X_test5, y_train5, y_test5 = make_split(X_balanced_encoded, y_balanced, random_state + 1)
X_train6, X_test6, y_train6, y_test6 = make_split(X_balanced_encoded, y_balanced, random_state + 2)
print(pd.Series(y_train4).value_counts())
print(pd.Series(y_train5).value_counts())
print(pd.Series(y_train6).value_counts())

'Data Shapley: train4'
model = 'logistic'
problem = 'classification'
num_test = 500
directory_aids4 = './temp_aids4'
shapley_aids4 = DShap(X_train4, y_train4, X_test4, y_test4, num_test, sources=None,
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
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_AIDS4_Low_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_AIDS4,
          'G Shapley': G_Shap_AIDS4,
          'LOO':         LOO_AIDS4},
    X_train=X_train4, y_train=y_train4, X_test=X_test4,   y_test=y_test4,
    a=0.05, b=0.4, ylim=(66, 72), order='ascending', random_iters=10, save_path=path)

'Save results'
train4_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS/train4.csv"
test4_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS//test4.csv"
train4_df = pd.DataFrame(X_train4, columns=[f"feature_{i}" for i in range(X_train4.shape[1])])
train4_df["label"] = y_train4
train4_df["TMC_Shap"] = TMC_Shap_AIDS4
train4_df["G_Shap"] = G_Shap_AIDS4
train4_df["LOO"] = LOO_AIDS4
train4_df.to_csv(train4_path, index=False)

test4_df = pd.DataFrame(X_test4, columns=[f"feature_{i}" for i in range(X_test4.shape[1])])
test4_df["label"] = y_test4
test4_df.to_csv(test4_path, index=False)

'Data Shapley: train5'
model = 'logistic'
problem = 'classification'
num_test = 500
directory_aids5 = './temp_aids5'
shapley_aids5 = DShap(X_train5, y_train5, X_test5, y_test5, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids4, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids5.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS5 = shapley_aids5.vals_tmc
G_Shap_AIDS5 = shapley_aids5.vals_g
LOO_AIDS5 = shapley_aids5.vals_loo
increments_AIDS5 = np.arange(0, len(X_train5)+1, 10)
rand_AIDS5 = np.random.rand(len(X_train5))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_AIDS5_High_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_AIDS5,
          'G Shapley': G_Shap_AIDS5,
          'LOO':         LOO_AIDS5},
    X_train=X_train5, y_train=y_train5, X_test=X_test5, y_test=y_test5,
    a=0.05, b=0.4, ylim=(55, 85), order='descending', random_iters=10, save_path=path)

'Save results'
train5_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS/train5.csv"
test5_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS//test5.csv"
train5_df = pd.DataFrame(X_train5, columns=[f"feature_{i}" for i in range(X_train5.shape[1])])
train5_df["label"] = y_train5
train5_df["TMC_Shap"] = TMC_Shap_AIDS5
train5_df["G_Shap"] = G_Shap_AIDS5
train5_df["LOO"] = LOO_AIDS5
train5_df.to_csv(train5_path, index=False)

test5_df = pd.DataFrame(X_test5, columns=[f"feature_{i}" for i in range(X_test5.shape[1])])
test5_df["label"] = y_test5
test5_df.to_csv(test5_path, index=False)


'Data Shapley: train5'
model = 'logistic'
problem = 'classification'
num_test = 500
directory_aids6 = './temp_aids6'
shapley_aids6 = DShap(X_train6, y_train6, X_test6, y_test6, num_test, sources=None,
                   sample_weight=None, model_family=model, metric='accuracy',
                   overwrite=True, directory=directory_aids6, seed=250421,
                   max_iter=500, tol=1e-20)
shapley_aids6.run(save_every=100, err=0.01, g_run=True, loo_run=True)
TMC_Shap_AIDS6 = shapley_aids6.vals_tmc
G_Shap_AIDS6 = shapley_aids6.vals_g
LOO_AIDS6 = shapley_aids6.vals_loo
increments_AIDS6 = np.arange(0, len(X_train6)+1, 10)
rand_AIDS6 = np.random.rand(len(X_train6))


### ----- 1. Performance plot (removal based on descending Data Shapley values) ----- ###
path = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515/Removal_AIDS6_Low_mean(0.01_500).png'
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_Shap_AIDS6,
          'G Shapley': G_Shap_AIDS6,
          'LOO':         LOO_AIDS6},
    X_train=X_train6, y_train=y_train6, X_test=X_test6, y_test=y_test6,
    a=0.05, b=0.4, ylim=(65, 88), order='ascending', random_iters=10, save_path=path)

'Save results'
train6_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS/train6.csv"
test6_path = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS//test6.csv"
train6_df = pd.DataFrame(X_train6, columns=[f"feature_{i}" for i in range(X_train6.shape[1])])
train6_df["label"] = y_train6
train6_df["TMC_Shap"] = TMC_Shap_AIDS6
train6_df["G_Shap"] = G_Shap_AIDS6
train6_df["LOO"] = LOO_AIDS6
train6_df.to_csv(train6_path, index=False)

test6_df = pd.DataFrame(X_test6, columns=[f"feature_{i}" for i in range(X_test6.shape[1])])
test6_df["label"] = y_test6
test6_df.to_csv(test6_path, index=False)