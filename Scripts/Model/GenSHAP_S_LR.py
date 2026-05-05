''' Model architecture of this verison: 6 GIN branches // 6 Transformer branches'''
'''Extra features are processed in final prediction head.'''
'''
0. Import Packages
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import re
import dgl
import torch
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from itertools import cycle
from dgl.nn.pytorch.conv import GINConv
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.nn import (Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer, LayerNorm)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal

'Check Availability of GPU'
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

## 'On CPU'
# device = torch.device("cpu")
# print("CUDA available:", torch.cuda.is_available())
# print("Using device:", device)
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda" if USE_CUDA else "cpu")
#
# print("CUDA available:", USE_CUDA)
# if USE_CUDA:
#     try:
#         print("Current device:", torch.cuda.current_device())
#         print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     except Exception as e:
#         print("CUDA info not available:", e)

'''
1. SVD
'''
'Flip Sign'
def flip_pe_sign(x, method='sum'):
    assert method in ['none', 'pos', 'sum', 'l1', 'l2', 'linf', 'random']
    if method == 'none':
        return x
    elif method == 'random':
        sign = torch.randint(0, 2, (x.shape[1],))*2 - 1
        return x * sign
    fliped_x = -1 * x
    if method == 'pos':
        x_v = torch.sum(x >= 0, dim=0)
        fliped_x_v = torch.sum(fliped_x >= 0, dim=0)
    elif method == 'sum':
        x_v = torch.sum(x, dim=0)
        fliped_x_v = torch.sum(fliped_x, dim=0)
    elif method == 'linf':
        x_v, _ = torch.max(x, dim=0)
        fliped_x_v, _ = torch.max(fliped_x, dim=0)
    elif method == 'l1':
        x_pos_mask = x >= 0
        x_v = torch.norm(x * x_pos_mask, dim=0, p=1)
        fliped_x_pos_mask = fliped_x >= 0
        fliped_x_v = torch.norm(fliped_x * fliped_x_pos_mask, dim=0, p=1)
    elif method == 'l2':
        x_pos_mask = x >= 0
        x_v = torch.norm(x * x_pos_mask, dim=0, p=2)
        fliped_x_pos_mask = fliped_x >= 0
        fliped_x_v = torch.norm(fliped_x * fliped_x_pos_mask, dim=0, p=2)
    else:
        raise NotImplementedError
    diff = x_v - fliped_x_v
    sign = torch.sign(diff)
    return x * sign

'SVD'
svd_d = 100
FEAT_DIM = svd_d
EXTRA_DIM = 50 + 100
# 1. Gradient Norm + Grad Error (2)
# 2. KNN Ratios (7)
# 3. Loss per 10 iterations (21)
# 4. differences on loss per 10 iterations (20)
# 5. samples of marginal contributions (100)

def SVD_Transform(A, d = svd_d):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(d, U.shape[1])
    X_svd = U[:, :k] * np.sqrt(S[:k])
    if k < d:
        pad = np.zeros((X_svd.shape[0], d - k), dtype = X_svd.dtype)
        X_svd = np.concatenate([X_svd, pad], axis = 1)
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = flip_pe_sign(X_t, method='sum')
    return X_t.numpy()

'''
2. Graph Construction
'''
C_List = [0.25, 0.5 , 1, 2, 4]

# def build_graph(X_raw: np.ndarray, y_raw: np.ndarray, phi: np.ndarray,
#                 total_util: float, c: float, *, D: np.ndarray, sigma: float,
#                 grad_norm_z: np.ndarray, grad_err_z: np.ndarray,
#                 knn_ratio_z: np.ndarray,
#                 loss_path_z: np.ndarray, loss_diff_z: np.ndarray,
#                 mc_margin_z: np.ndarray,
#                 d: int = svd_d, weight_extra: float = 3.0) -> Data:
#     ## Gaussian Kernel Adjacency
#     A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2)).astype(np.float32, copy = False)
#     X_svd = SVD_Transform(A, d = d).astype(np.float32, copy = False)
#     y = np.asarray(y_raw).ravel()
#     A_y = (y[:, None] == y[None, :]).astype(np.float32, copy=False)
#
#     ## Injection of Gradient Norm + Grad Error + KNN ratio + final loss
#     grad_norm_z = np.asarray(grad_norm_z, dtype = np.float32).reshape(-1, 1)  # (N, 1)
#     grad_err_z = np.asarray(grad_err_z, dtype = np.float32).reshape(-1, 1)  # (N, 1)
#     knn_ratio_z = np.asarray(knn_ratio_z, dtype = np.float32)   # (N, 7)
#     loss_path_z = np.asarray(loss_path_z, dtype = np.float32)   # (N, 21)
#     loss_diff_z = np.asarray(loss_diff_z, dtype = np.float32)   # (N, 20)
#     mc_margin_z = np.asarray(mc_margin_z, dtype = np.float32)   # (N, 6)
#     ## Weighted Injected Features
#     extra_feats = np.concatenate([grad_norm_z, grad_err_z, knn_ratio_z,
#                                   loss_path_z, loss_diff_z, mc_margin_z], axis=1)  # (N, 56)
#     extra_feats = (weight_extra * extra_feats).astype(np.float32, copy=False)
#
#     ## Fully connected graph
#     N = X_raw.shape[0]
#     src, dst = np.triu_indices(N, k=1)
#     eidx = np.vstack([src, dst])
#     edge_index = torch.tensor(np.concatenate([eidx, eidx[::-1]], axis=1), dtype=torch.long, device = device)
#
#     w_feat = A[src, dst]
#     w_label = A_y[src, dst]
#     edge_attr_feat = torch.tensor(np.concatenate([w_feat, w_feat]), dtype=torch.float32, device = device).unsqueeze(1)
#     edge_attr_label = torch.tensor(np.concatenate([w_label, w_label]), dtype=torch.float32, device = device).unsqueeze(1)
#
#     data = Data(
#         x=torch.tensor(X_svd, dtype=torch.float32, device = device),
#         edge_index=edge_index,
#         edge_attr_feat=edge_attr_feat,
#         edge_attr_label=edge_attr_label
#     )
#
#     ## Attributes
#     data.phi = torch.tensor(phi, dtype=torch.float32, device = device)
#     data.total_util = torch.tensor(total_util, dtype=torch.float32, device = device)
#     data.y = torch.tensor(y_raw, dtype=torch.long, device = device)
#     data.extra = torch.tensor(extra_feats, dtype = torch.float32, device = device)
#
#     ## Record number of classes of each dataset
#     K = int(np.unique(y).size)
#     k_vec = np.array([K, 1/max(1e-12, K), np.log(K)], dtype=np.float32)    # (N, 3)
#     data.k_feat = torch.tensor(np.tile(k_vec, (N, 1)), dtype=torch.float32, device = device)
#
#     ## Inject total utility
#     u = float(total_util)
#     u_vec = np.array([u, 1.0/max(u, 1e-12)], dtype = np.float32)  # (2,)
#     data.u_feat = torch.tensor(np.tile(u_vec, (N, 1)), dtype=torch.float32, device = device)
#
#     ## Keep grad_norm_z, grad_err_z for diagnostics
#     data.grad_norm_z = torch.tensor(grad_norm_z, dtype=torch.float32, device = device)
#     data.grad_err_z  = torch.tensor(grad_err_z, dtype=torch.float32, device = device)
#     data.knn_ratio_z = torch.tensor(knn_ratio_z, dtype=torch.float32, device = device)
#
#     return data

## To fix memory limits, move one dataset to GPU
def build_graph(X_raw: np.ndarray, y_raw: np.ndarray, phi: np.ndarray,
                total_util: float, c: float, *, D: np.ndarray, sigma: float,
                grad_norm_z: np.ndarray, grad_err_z: np.ndarray,
                knn_ratio_z: np.ndarray,
                loss_path_z: np.ndarray, loss_diff_z: np.ndarray,
                mc_margin_z: np.ndarray,
                d: int = svd_d, weight_extra: float = 3.0) -> Data:
    ## Gaussian Kernel Adjacency
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2)).astype(np.float32, copy=False)
    X_svd = SVD_Transform(A, d=d).astype(np.float32, copy=False)
    y = np.asarray(y_raw).ravel()
    A_y = (y[:, None] == y[None, :]).astype(np.float32, copy=False)

    ## Injection of Gradient Norm + Grad Error + KNN ratio + final loss
    grad_norm_z = np.asarray(grad_norm_z, dtype=np.float32).reshape(-1, 1)
    grad_err_z = np.asarray(grad_err_z, dtype=np.float32).reshape(-1, 1)
    knn_ratio_z = np.asarray(knn_ratio_z, dtype=np.float32)
    loss_path_z = np.asarray(loss_path_z, dtype=np.float32)
    loss_diff_z = np.asarray(loss_diff_z, dtype=np.float32)
    mc_margin_z = np.asarray(mc_margin_z, dtype=np.float32)

    ## Weighted Injected Features
    extra_feats = np.concatenate(
        [grad_norm_z, grad_err_z, knn_ratio_z, loss_path_z, loss_diff_z, mc_margin_z],
        axis=1
    ).astype(np.float32, copy=False)
    extra_feats = (weight_extra * extra_feats).astype(np.float32, copy=False)

    ## Fully connected graph
    N = X_raw.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(
        np.concatenate([eidx, eidx[::-1]], axis=1),
        dtype=torch.long
    )  # CPU only

    w_feat = A[src, dst]
    w_label = A_y[src, dst]
    edge_attr_feat = torch.tensor(
        np.concatenate([w_feat, w_feat]),
        dtype=torch.float32
    ).unsqueeze(1)  # CPU only
    edge_attr_label = torch.tensor(
        np.concatenate([w_label, w_label]),
        dtype=torch.float32
    ).unsqueeze(1)  # CPU only

    data = Data(
        x=torch.tensor(X_svd, dtype=torch.float32),   # CPU only
        edge_index=edge_index,
        edge_attr_feat=edge_attr_feat,
        edge_attr_label=edge_attr_label
    )

    ## Attributes
    data.phi = torch.tensor(phi, dtype=torch.float32)
    data.total_util = torch.tensor(total_util, dtype=torch.float32)
    data.y = torch.tensor(y_raw, dtype=torch.long)
    data.extra = torch.tensor(extra_feats, dtype=torch.float32)

    ## Record number of classes of each dataset
    K = int(np.unique(y).size)
    k_vec = np.array([K, 1 / max(1e-12, K), np.log(K)], dtype=np.float32)
    data.k_feat = torch.tensor(np.tile(k_vec, (N, 1)), dtype=torch.float32)

    ## Inject total utility
    u = float(total_util)
    u_vec = np.array([u, 1.0 / max(u, 1e-12)], dtype=np.float32)
    data.u_feat = torch.tensor(np.tile(u_vec, (N, 1)), dtype=torch.float32)

    ## Keep grad_norm_z, grad_err_z for diagnostics
    data.grad_norm_z = torch.tensor(grad_norm_z, dtype=torch.float32)
    data.grad_err_z  = torch.tensor(grad_err_z, dtype=torch.float32)
    data.knn_ratio_z = torch.tensor(knn_ratio_z, dtype=torch.float32)

    return data

'''
3. Data Preprocessing
'''
def _safe_lr_utility(X_tr, y_tr, X_te, y_te, max_iter=200, random_state=23):
    """
    Utility = test accuracy of logistic regression.
    Returns np.nan if the subset is not trainable (e.g. only one class).
    """
    y_tr = np.asarray(y_tr).ravel()
    if len(np.unique(y_tr)) < 2:
        return np.nan

    try:
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
        clf.fit(X_tr, y_tr)
        return float(clf.score(X_te, y_te))
    except Exception:
        return np.nan


def compute_mc_marginal_features(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 20,
    random_state: int = 23,
    min_subset_size: int = 5,
    max_subset_size: int = None,
    max_tries_per_sample: int = 30,
) -> np.ndarray:
    """
    For each data point i, sample subsets S not containing i and compute:
        U(S ∪ {i}) - U(S)
    Then summarize the sampled marginal contributions into 6 features:
        [mean, std, pos_rate, neg_rate, min, max]

    Output shape: (N, 6)
    """
    rng = np.random.default_rng(random_state)

    X_raw = np.asarray(X_raw)
    y_raw = np.asarray(y_raw).ravel()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).ravel()

    N = X_raw.shape[0]
    K = len(np.unique(y_raw))
    if max_subset_size is None:
        max_subset_size = max(min(N - 1, max(min_subset_size + 1, N // 2)), min_subset_size)

    feats = np.zeros((N, n_samples), dtype=np.float32)

    all_idx = np.arange(N)

    for i in range(N):
        others = all_idx[all_idx != i]
        diffs = []

        for _ in range(n_samples):
            accepted = False

            for _try in range(max_tries_per_sample):
                k = int(rng.integers(min_subset_size, max_subset_size + 1))
                if k > len(others):
                    continue

                S = rng.choice(others, size=k, replace=False)
                yS = y_raw[S]

                # need at least 2 classes in S so LR is trainable
                if len(np.unique(yS)) < min(2, K):
                    continue

                u_S = _safe_lr_utility(X_raw[S], y_raw[S], X_test, y_test)
                if np.isnan(u_S):
                    continue

                S_plus_i = np.concatenate([S, np.array([i])])
                u_Si = _safe_lr_utility(X_raw[S_plus_i], y_raw[S_plus_i], X_test, y_test)
                if np.isnan(u_Si):
                    continue

                diffs.append(u_Si - u_S)
                accepted = True
                break

            if not accepted:
                # fallback if no valid subset found
                diffs.append(0.0)

        feats[i, :] = np.asarray(diffs, dtype = np.float32)

    return feats

def load_and_preprocess(train_csv: str, test_csv: str, d: int = svd_d):
    def _standardize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype = np.float64)
        mu = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return np.zeros_like(arr, dtype = np.float32)
        return ((arr - mu) / std).astype(np.float32)

    ## Read-in train-test pairs
    df_train = pd.read_csv(train_csv)
    X_raw = df_train.drop(columns=['target','PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy().astype(np.float32)

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy().astype(int)

    ## Compute distance
    D = pairwise_distances(X_raw, metric='euclidean').astype(np.float32, copy = False)
    sigma = max(float(np.mean(D)), 1e-15)

    ## Compute KNN Ratio
    K_List = [5, 7, 9, 10, 15, 20, 30]
    max_k = max(K_List)
    D_no_self = D.copy()
    np.fill_diagonal(D_no_self, np.inf)
    nn_idx = np.argpartition(D_no_self, kth=max_k - 1, axis=1)[:, :max_k]
    rows = np.arange(D.shape[0])[:, None]
    order = np.argsort(D_no_self[rows, nn_idx], axis=1)
    nn_idx = nn_idx[rows, order]
    y = np.asarray(y_raw).ravel()
    neigh_y = y[nn_idx]
    diff = (neigh_y != y[:, None])

    knn_ratios = [diff[:, :k].mean(axis=1) for k in K_List]
    knn_ratio_feat = np.stack(knn_ratios, axis=1)
    knn_ratio_z = np.column_stack([_standardize(knn_ratio_feat[:, i]) for i in range(knn_ratio_feat.shape[1])]).astype(np.float32, copy = False)

    ## Algorithm: Logistic Regression
    lr_full = LogisticRegression(max_iter = 200, random_state=23)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    ## Compute gradient norm and gradient error
    P = lr_full.predict_proba(X_raw)  # (N, C)
    classes = lr_full.classes_  # shape (C,)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    n = X_raw.shape[0]
    col_idx = np.array([class_to_idx[yy] for yy in y_raw], dtype = int)  # (N,)
    Y_onehot = np.zeros_like(P)
    Y_onehot[np.arange(n), col_idx] = 1.0

    grad_w = (P - Y_onehot)[:, :, None] * X_raw[:, None, :]  # (N, C, d_feat)
    grad_norm = np.linalg.norm(grad_w, axis=(1, 2))  # (N,)
    p_true = P[np.arange(n), col_idx]  # (N,)
    grad_err = 1.0 - p_true

    grad_norm_z = _standardize(grad_norm)
    grad_err_z  = _standardize(grad_err)

    ## Compute training loss at 1st iter, last iter, and difference
    iter_list = [1] + list(range(10, 201, 10))
    loss_path = []
    lr_1 = LogisticRegression(max_iter = 1, random_state = 23)
    lr_1.fit(X_raw, y_raw)
    P_1 = lr_1.predict_proba(X_raw)
    p_true_1 = np.clip(P_1[np.arange(n), col_idx], 1e-12, 1)
    point_loss_1 = -np.log(p_true_1)
    loss_path.append(point_loss_1)

    for it in range(10, 201, 10):
        lr_it = LogisticRegression(max_iter = it, random_state = 23)
        lr_it.fit(X_raw, y_raw)
        P_it = lr_it.predict_proba(X_raw)
        p_true_it = np.clip(P_it[np.arange(n), col_idx], 1e-12, 1)
        point_loss_it = -np.log(p_true_it)
        loss_path.append(point_loss_it)

    loss_path = np.stack(loss_path, axis = 1)
    loss_diff = loss_path[:, :-1] - loss_path[:, 1:]
    loss_path_z = np.column_stack([_standardize(loss_path[:, j]) for j in range(loss_path.shape[1])]).astype(np.float32, copy = False)
    loss_diff_z = np.column_stack([_standardize(loss_diff[:, j]) for j in range(loss_diff.shape[1])]).astype(np.float32, copy = False)

    ## Compute statistics of marginal contribution
    mc_margin_feat = compute_mc_marginal_features(
        X_raw, y_raw, X_test, y_test,
        n_samples = 100, random_state = 23,
        min_subset_size = max(5, len(np.unique(y_raw)) + 1),
        max_subset_size = min(X_raw.shape[0] - 1, 50)
    )
    mc_margin_z = np.column_stack([_standardize(mc_margin_feat[:, j]) for j in range(mc_margin_feat.shape[1])]).astype(np.float32, copy = False)
    # ## Compute LOO values
    # n = X_raw.shape[0]
    # loo_raw = np.zeros(n, dtype=np.float64)
    #
    # for i in range(n):
    #     mask = np.ones(n, dtype=bool)
    #     mask[i] = False
    #     lr_i = LogisticRegression(max_iter=200, random_state=23)
    #     lr_i.fit(X_raw[mask], y_raw[mask])
    #     acc_i = lr_i.score(X_test, y_test)
    #     loo_raw[i] = model_acc - acc_i  # LOO_i = U(D) - U(D\{i})
    #
    # loo_z = _standardize(loo_raw)

    ## build graphs on device
    data_list = []
    for c in C_List:
        data = build_graph(
            X_raw, y_raw, phi, total_util, c,
            D = D, sigma = sigma,
            grad_norm_z = grad_norm_z,
            grad_err_z = grad_err_z,
            knn_ratio_z = knn_ratio_z,
            loss_path_z = loss_path_z,
            loss_diff_z = loss_diff_z,
            mc_margin_z = mc_margin_z,
            d=d
        )
        data_list.append(data)

    return data_list

'''
4. Model Construction
'''
' Version A. Sequantial GIN & GT '
# class GINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels = 256, out_channels = 1, heads = 8):
#         super().__init__()
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))
#
#         self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.branches = torch.nn.ModuleList()
#         for _ in C_List:
#             convs = torch.nn.ModuleList()
#             for layer in range(3):
#                 convs.append(
#                     GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
#                             aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
#                             activation = ReLU())
#                 )
#             self.branches.append(convs)
#
#         self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))
#
#         d_concat = hidden_channels * len(C_List)
#         encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
#         self.gt = TransformerEncoder(encoder_layer, num_layers = 3)
#
#         self.Z = Sequential(Linear(d_concat, hidden_channels), ReLU())
#         self.out = Sequential(
#             Linear(hidden_channels + 3, hidden_channels * 2),
#             ReLU(),
#             Linear(hidden_channels * 2, 100),
#             ReLU(),
#             Linear(100, out_channels)
#         )
#
#     def forward(self, data_list):
#         data = data_list[0].to(device)
#         x0 = data.x
#         output_GIN = []
#         for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
#             node_input = gdata.x.to(device)
#             edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)
#             edge_attr_label = gdata.edge_attr_label.view(-1).to(device)
#
#             weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
#             edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label
#
#             g = dgl.graph(
#                 (gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
#                 num_nodes=node_input.size(0),
#                 device=device
#             )
#             g.edata['w'] = edge_weight
#
#             for conv in convs:
#                 node_input = conv(g, node_input, edge_weight=g.edata['w'])
#             output_GIN.append(node_input)
#
#         output_GIN = torch.cat(output_GIN, dim = 1)
#         output_skip = output_GIN + self.skip_lin(x0)
#
#         output_skip = output_skip.unsqueeze(0)  # (1, N, d_concat)
#         output_gt = self.gt(output_skip).squeeze(0)
#
#         output_h = self.Z(output_gt)
#         k_col = data.k_feat
#         h_plus = torch.cat([output_h, data.k_feat], dim = 1)
#         return self.out(h_plus)

' Version B. Parallel GIN & GT '
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels = 64, out_channels = 1, heads = 2,
                 gin_layers = 2, gt_layers = 2):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))

        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(gin_layers):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                            aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
                            activation = ReLU())
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List)).to(device)
        d_concat = hidden_channels * len(C_List) # 64 * 6 = 384

        self.Z_gin = Sequential(
            Linear(d_concat, hidden_channels),
            ReLU()
        ).to(device)

        ### Transformer block
        assert in_channels % heads == 0, "gt_d_model should be a multiple of heads"
        self.gt_branches = torch.nn.ModuleList()
        self.Z_gt_branches = torch.nn.ModuleList()

        for _ in C_List:
            encoder_layer = TransformerEncoderLayer(d_model = in_channels, nhead = heads,
                                                    dim_feedforward = 1024,
                                                    dropout = 0.0, batch_first = True)
            gt_branch = TransformerEncoder(encoder_layer, num_layers = gt_layers)
            self.gt_branches.append(gt_branch)

            self.Z_gt_branches.append(Sequential(Linear(in_channels, hidden_channels), ReLU()))

        self.fuse_gt = Linear(hidden_channels * len(C_List), hidden_channels).to(device)

        extra_dim = EXTRA_DIM
        self.out = Sequential(
            Linear((hidden_channels * 2) + extra_dim + 3 + 2, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, 100),
            ReLU(),
            Linear(100, out_channels)
        ).to(device)

    def forward(self, data_list):
        c1_idx = C_List.index(1)
        data1 = data_list[c1_idx]
        N = data1.x.size(0)

        ### GIN Branches
        output_GIN = []
        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x
            edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label

            g = dgl.graph(
                (gdata.edge_index[0], gdata.edge_index[1]),
                num_nodes = N, device = device
            )
            g.edata["w"] = edge_weight

            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])

            output_GIN.append(node_input)

        gin_cat = torch.cat(output_GIN, dim=1)  # (N, 384) when hidden_channels = 64
        gin_cat = gin_cat + self.skip_lin(data1.x)  # (N, 384) when hidden_channels = 64
        h_gin = self.Z_gin(gin_cat)  # (N, 64) when hidden_channels = 64

        ### 6 Transformer branches (shared weights, each corresponding to c in C_List)
        h_gt_list = []
        for gt_branch, z_branch, gdata in zip(self.gt_branches, self.Z_gt_branches, data_list):
            x = gdata.x.unsqueeze(0) # (1, N, in_channels)
            gt_out = gt_branch(x).squeeze(0) # (N, in_channels)
            h_gt_i = z_branch(gt_out)  # (N, hidden_channels)
            h_gt_list.append(h_gt_i)

        h_gt_cat = torch.cat(h_gt_list, dim=1)  # (N, 64*6=384)
        h_gt = self.fuse_gt(h_gt_cat)  # (N, 64)

        h_fused = torch.cat([h_gin, h_gt], dim=1)  # (N, 128)
        h_plus = torch.cat([h_fused, data1.extra, data1.k_feat, data1.u_feat], dim=1)  # (N, 183)
        return self.out(h_plus)

'''
5. Training (standard scheme)
'''
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def discover_datasets(base_dir: str, require_tmc: bool = False):
    TEST_RE = re.compile(r"^(.+?)_test_(\d+)(?:\.[^.]+)?$")
    TMC_RE = re.compile(r"^(.+?)_TMC_(\d+)(?:\.[^.]+)?$")
    TRAIN_RE = re.compile(r"^(.+?)_(\d+)(?:\.[^.]+)?$")

    bucket = {}
    for fn in os.listdir(base_dir):
        base = fn
        path = os.path.join(base_dir, fn)

        m = TEST_RE.match(base)
        if m:
            key = (m.group(1), m.group(2))
            bucket.setdefault(key, {})["test"] = path
            continue

        m = TMC_RE.match(base)
        if m:
            key = (m.group(1), m.group(2))
            bucket.setdefault(key, {})["tmc"] = path
            continue

        if TEST_RE.match(base) or TMC_RE.match(base):
            continue
        m = TRAIN_RE.match(base)
        if m:
            key = (m.group(1), m.group(2))
            bucket.setdefault(key, {})["train_like"] = path
            continue

    result = []
    for key, paths in bucket.items():
        tr = paths.get("train_like"); te = paths.get("test"); tm = paths.get("tmc")
        if tr and te and (tm or not require_tmc):
            stem, idx = key
            base_name = f"{stem}_{idx}"
            result.append({
                "key": key,
                "base_name": base_name,
                "train_like": tr,
                "test": te,
                "tmc": tm
            })
        else:
            stem, idx = key
            want = []
            if not paths.get("train_like"): want.append("train_like")
            if not paths.get("test"):       want.append("test")
            if require_tmc and not paths.get("tmc"): want.append("tmc")
            print(f"WARNING: Incomplete set for {stem}_{idx}: missing {', '.join(want)}")
    return result

def KDE_weights(y: torch.Tensor, sigma: float, eps: float = 1e-12) -> torch.Tensor:
    y = y.view(-1)
    N = y.numel()
    sigma = float(max(sigma, eps))

    diff2 = (y[:, None] - y[None, :]) ** 2
    K = torch.exp(-diff2 / (2.0 * sigma * sigma)) / (np.sqrt(2.0 * np.pi) * sigma)
    rho = K.mean(dim = 1)
    w = rho/(rho.sum() + eps)
    return w

def train(model, dataset_graphs, *, epochs=500, lr=1e-5, weight_decay=1e-7,
          base_test_dir: str = None, holdout_bases: list = None, out_dir: str = None,
          scheduler_type: str = "step", step_size: int = 50, step_gamma: float = 0.5,
          lambda_fn = None, eval_every: int = 50, c_w = 0.5,
          train_graph_build_sec: float = 0.0, train_graph_build_sec_by_dataset = None):

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type is not None:
        stype = scheduler_type.lower()
        if stype == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=step_gamma)
        elif stype == "lambda":
            if lambda_fn is None:
                lambda_fn = (lambda ep: 0.5*(1 + np.cos(np.pi*ep / max(1, epochs))))
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
        elif stype == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    if out_dir:
        _ensure_dir(out_dir)

    ## Cache holdouts
    holdout_cache = {}
    holdout_pred_sec_sum = {}
    holdout_pred_count = {}
    holdout_build_sec = {}

    catalog = None
    if base_test_dir and holdout_bases:
        catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc=False)}

    if catalog is not None and holdout_bases:
        for base in holdout_bases:
            p = catalog.get(base, None)
            if p is None:
                print(f"WARNING: holdout base {base} not found in catalog!")
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dl = load_and_preprocess(p["train_like"], p["test"], d = svd_d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            build_sec = time.perf_counter() - t0
            holdout_build_sec[base] = float(build_sec)
            print(f"[holdout cached] {base}: build_sec = {build_sec: .6f}")

            holdout_cache[base] = dl
            holdout_pred_sec_sum[base] = 0.0
            holdout_pred_count[base] = 0

    ## Train
    rows = []
    total_train_sec = 0.0

    for ep in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_train = time.perf_counter()

        model.train()
        wloss_list = []
        mse_list = []

        # for data_list in dataset_graphs:
        #     total_util = float(data_list[0].total_util.item())
        #     phi_true = data_list[0].phi.view(-1)
        #     N = phi_true.numel()
        #
        #     scale = N/max(total_util, 1e-15)
        #     phi_true_scaled = phi_true * scale
        #     phi_hat_scaled = model(data_list).view(-1)

        ## move one dataset to GPU
        for data_list_cpu in dataset_graphs:
            data_list = to_device_datalist(data_list_cpu, device)

            total_util = float(data_list[0].total_util.item())
            phi_true = data_list[0].phi.view(-1)
            N = phi_true.numel()

            scale = N / max(total_util, 1e-15)
            phi_true_scaled = phi_true * scale
            phi_hat_scaled = model(data_list).view(-1)


            ## Record MSE
            phi_hat = phi_hat_scaled * (total_util / N)
            mse_phi = F.mse_loss(phi_hat, phi_true, reduction = "mean")

            # ## Training with weighted MSE (exp(-y_i/c)(\hat{y_i}-y_i)^2):
            # w = torch.exp(-phi_true_scaled / c_w)
            # w = torch.clamp(w, 0.1, 10.0)
            # wloss = torch.mean(w * (phi_hat_scaled - phi_true_scaled) ** 2)

            # ## Training with KDE Weighted MSE:
            # sigma_kde = float(phi_true_scaled.mean().item())
            # sigma_kde = max(sigma_kde, 1e-12)
            # w = KDE_weights(phi_true_scaled, sigma=sigma_kde)
            # wloss = torch.sum(w * (phi_hat_scaled - phi_true_scaled) ** 2)

            ## Training with KDE Weighted MSE(sigma = mean of pairwise distance):
            y1 = phi_true_scaled.view(-1, 1)
            if y1.size(0) >= 2:
                sigma_kde = float(torch.pdist(y1, p = 2).mean().item())
            else:
                sigma_kde = 1e-12
            sigma_kde = max(sigma_kde, 1e-12)
            w = KDE_weights(phi_true_scaled, sigma=sigma_kde)
            wloss = torch.sum(w * (phi_hat_scaled - phi_true_scaled) ** 2)

            optimizer.zero_grad(set_to_none=True)
            wloss.backward()
            optimizer.step()
            wloss_list.append(wloss.detach())
            mse_list.append(mse_phi.detach())

            del data_list


        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0_train
        total_train_sec += train_sec

        if scheduler is not None:
            scheduler.step()

        train_wloss = float(torch.stack(wloss_list).mean().item())
        train_mse = float(torch.stack(mse_list).mean().item())
        row = {"epoch": ep, "weighted_loss": train_wloss, "train_mse": train_mse}

        ## Evaluation
        do_eval = (holdout_cache is not None) and (len(holdout_cache) > 0) and (ep % eval_every == 0)
        if do_eval:
            model.eval()
            mses = []
            with torch.no_grad():
                # for base, dl in holdout_cache.items():
                #     if device.type == "cuda":
                #         torch.cuda.synchronize()
                #     t0_pred = time.perf_counter()
                #
                #     pred_scaled = model(dl).view(-1)
                #     phi_t = dl[0].phi.view(-1)
                #     total_util_t = dl[0].total_util
                #     N_t = phi_t.numel()
                #
                #     pred_phi = pred_scaled * (total_util_t / N_t)
                #     m = float(F.mse_loss(pred_phi, phi_t, reduction="mean").item())

                for base, dl_cpu in holdout_cache.items():
                    dl = to_device_datalist(dl_cpu, device)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0_pred = time.perf_counter()

                    pred_scaled = model(dl).view(-1)
                    phi_t = dl[0].phi.view(-1)
                    total_util_t = dl[0].total_util
                    N_t = phi_t.numel()

                    pred_phi = pred_scaled * (total_util_t / N_t)
                    m = float(F.mse_loss(pred_phi, phi_t, reduction="mean").item())

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    pred_sec = time.perf_counter() - t0_pred

                    holdout_pred_sec_sum[base] += float(pred_sec)
                    holdout_pred_count[base] += 1

                    row[base] = m
                    mses.append(m)

                    del dl

            row["avg_mse"] = float(np.mean(mses)) if mses else np.nan
            print(f"Epoch {ep:5d} | wLoss: {train_wloss:.8e} | Train MSE: {train_mse:.8e} | Holdout(avg): {row['avg_mse']:.8e}")
        else:
            row["avg_mse"] = np.nan
            print(f"Epoch {ep:5d} | wLoss: {train_wloss:.8e} | Train MSE: {train_mse:.8e}")
        rows.append(row)

    if out_dir:
        df = pd.DataFrame(rows)

        ## Save 1) epoch; 2) train loss; 3) train MSE; 4) test MSE(per 50 epochs)
        if holdout_bases:
            for b in holdout_bases:
                if b not in df.columns:
                    df[b] = np.nan

            mse_csv = os.path.join(out_dir, "Training_MSE.csv")
            keep_cols = ["epoch", "weighted_loss", "train_mse"]
            if holdout_bases:
                keep_cols += [b for b in holdout_bases if b in df.columns]
            keep_cols += ["avg_mse"]
            df[keep_cols].to_csv(mse_csv, index=False)
            print("saved ➜", mse_csv)

        total_genshap_s_train_sec = float(train_graph_build_sec + total_train_sec)
        train_graph_build_sec_by_dataset = train_graph_build_sec_by_dataset or {}

        # Save: GenSHAP-S training time; sampled marginal probe time is included
        # in the graph preprocessing measurement below.
        time_txt = os.path.join(out_dir, "Train_Test_Time.txt")
        with open(time_txt, "w") as f:
            f.write(f"total_train_sec: {total_genshap_s_train_sec:.6f}\n")
            f.write("total_train_sec_note: includes training graph preprocessing, including sampled marginal probe injection, plus epoch optimization; excludes final diagnostics.\n")
            f.write(f"train_graph_build_sec_including_sampling_marginal_probes: {train_graph_build_sec:.6f}\n")
            f.write(f"epoch_optimization_train_sec: {total_train_sec:.6f}\n")
            f.write(f"eval_every: {eval_every}\n\n")

            f.write("train_graph_build_time_sec (includes sampled marginal probes):\n")
            if not train_graph_build_sec_by_dataset:
                f.write("  (not recorded per dataset)\n")
            else:
                for base in train_graph_build_sec_by_dataset:
                    f.write(f"  {base}: build_sec={train_graph_build_sec_by_dataset[base]:.6f}\n")

            f.write("\nholdout_graph_build_time_sec:\n")
            if not holdout_build_sec:
                f.write("  (no holdout cached)\n")
            else:
                for base in holdout_build_sec:
                    f.write(f"  {base}: build_sec={holdout_build_sec[base]:.6f}\n")

            f.write("\nholdout_pred_time_sec (avg over evals):\n")
            if not holdout_pred_sec_sum:
                f.write("  (no holdout cached)\n")
            else:
                for base in holdout_pred_sec_sum:
                    cnt = max(1, holdout_pred_count.get(base, 0))
                    avg_pred = holdout_pred_sec_sum[base] / cnt
                    f.write(f"  {base}: pred_avg_sec={avg_pred:.6f}, eval_count={holdout_pred_count[base]}\n")
        print("saved ➜", time_txt)

        df_plot = df.copy()
        df_eval = df_plot[df_plot["avg_mse"].notna()].copy()
        plt.figure(figsize=(6, 4))
        plt.plot(df_plot["epoch"], df_plot["train_mse"], label="Train (MSE)", linewidth=1.2, alpha=0.9)
        if len(df_eval) > 0:
            plt.plot(df_eval["epoch"], df_eval["avg_mse"], label="Test(avg MSE)", linewidth=1.2, alpha=0.9)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Optimisation Curve of Training & Testing Sets (MSE)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Optimisation_Curve.png"))
        plt.close()

        # Log curve: plot the objective you actually optimize (weighted loss)
        plt.figure(figsize=(6, 4))
        plt.plot(
            df_plot["epoch"],
            np.log(df_plot["weighted_loss"].to_numpy() + 1e-12),
            label="Training Weighted Loss (log)",
            linewidth=1.2,
            alpha=0.9,
        )
        plt.xlabel("Epoch")
        plt.ylabel("log(loss)")
        plt.title("Optimisation Curve (log)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Optimisation_Curve_log.png"))
        plt.close()
    return rows

'''
6. Prediction
'''
def predict(model, data_list):
    model.eval()
    # with torch.no_grad():
    #     yhat_scaled = model(data_list).view(-1)
    #     total_util = data_list[0].total_util.to(yhat_scaled.device)
    #     N = data_list[0].x.size(0)
    #     descaled = yhat_scaled * (total_util / N)
    #     return descaled.detach().cpu().numpy()
    with torch.no_grad():
        data_list_dev = to_device_datalist(data_list, device)
        yhat_scaled = model(data_list_dev).view(-1)
        total_util = data_list_dev[0].total_util.to(yhat_scaled.device)
        N = data_list_dev[0].x.size(0)
        descaled = yhat_scaled * (total_util / N)
        out = descaled.detach().cpu().numpy()

        del data_list_dev

        return out

'''
7. Supplementary Functions
'''
def to_device_datalist(data_list, dev):
    out = []
    for g in data_list:
        gg = g.to(dev)
        out.append(gg)
    return out

def print_adj_weights(model):
    with torch.no_grad():
        print("\n=== Learned adjacency weights per c in C_List ===")
        print(" idx | c | alpha  beta  w_feature  w_label")
        for i, c in enumerate(C_List):
            a = model.alphas[i]; b = model.betas[i]
            probs = F.softmax(torch.stack([a, b]), dim = 0)
            print(f"{i:>4d} | {c:<6} | {a.item():<13.6f}  {b.item():<13.6f}  {probs[0].item():<16.6f}  {probs[1].item():<16.6f}")
        print()

COLORS = {"true": "tab:blue", "pred": "tab:purple", "tmc": "tab:orange"}

def _class_styles(unique_labels):
    FILLED = ['o','s','^','D','P','*','v','>','<','H']
    it = cycle(FILLED)
    styles = {}
    for cls in sorted(unique_labels):
        mk = next(it)
        styles[cls] = dict(marker=mk)
    return styles

def diagnose_train_sets(gnn, base_train_dir: str, out_dir: str, graph_dir: str = None):
    _ensure_dir(out_dir)
    graph_dir = graph_dir or out_dir
    _ensure_dir(graph_dir)

    triplets = discover_datasets(base_train_dir, require_tmc=False)
    for item in triplets:
        train_csv, test_csv = item["train_like"], item["test"]
        # dlist = load_and_preprocess(train_csv, test_csv, d=gnn.skip_lin.in_features)
        dlist = load_and_preprocess(train_csv, test_csv, d=svd_d)  # use svd_d
        preds = predict(gnn, dlist)

        df_train = pd.read_csv(train_csv)
        df_train['Pred_Shap'] = preds
        base_name = os.path.basename(train_csv).replace('.csv','')
        out_csv = os.path.join(out_dir, f'{base_name}_Pred.csv')
        df_train.to_csv(out_csv, index = False)
        print('Saved to:', out_csv)

        true_vals = df_train['PerDShap'].to_numpy()
        pred_vals = df_train['Pred_Shap'].to_numpy()
        y_labels = (df_train['target'].to_numpy().astype(int) if 'target' in df_train.columns else np.zeros_like(true_vals))
        mean_true, mean_pred = true_vals.mean(), pred_vals.mean()
        N = len(true_vals); idx = np.arange(N); width = 0.25

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

        # --- Bar plot ---
        bins = 50
        ax1.hist(true_vals, bins=bins, alpha=0.55, label='Permutated-based Data Shapley',
                 edgecolor='k', linewidth=0.3, color=COLORS["true"])
        ax1.hist(pred_vals, bins=bins, alpha=0.55, label='Predicted Data Shapley',
                 edgecolor='k', linewidth=0.3, color=COLORS["pred"])

        ax1.axvline(mean_true, linestyle='--', linewidth=2, color=COLORS["true"], label=f'Mean True: {mean_true:.3f}')
        ax1.axvline(mean_pred, linestyle='--', linewidth=2, color=COLORS["pred"], label=f'Mean Pred: {mean_pred:.3f}')

        ax1.set_xlabel('Data Shapley Value', fontsize=12, weight='bold')
        ax1.set_ylabel('Count', fontsize=12, weight='bold')
        ax1.set_title('Permutated-based vs Predicted Data Shapley (Histogram)', fontsize=13, weight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.legend(fontsize=10)

        # --- Scatter ---
        lo, hi = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
        pad = 0.005 * (hi - lo if hi > lo else 1.0)
        ax2.set_xlim(lo - pad, hi + pad)
        ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_aspect('equal', adjustable='box')

        classes = np.unique(y_labels)
        styles = _class_styles(classes)
        for c in classes:
            mask = (y_labels == c)
            st = styles[c].copy()
            ax2.scatter(true_vals[mask], pred_vals[mask],
                        **st, facecolors='none', edgecolors=COLORS["pred"], linewidths=1.2)

        ax2.plot([lo, hi], [lo, hi], '--k', linewidth=1.5)
        ax2.set_xlabel('Permutated-based Data Shapley Values', fontsize=12, weight='bold')
        ax2.set_ylabel('Predicted Data Shapley Values', fontsize=12, weight='bold')
        ax2.set_title('True vs Predicted (Scatter)', fontsize=13, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4)

        class_handles = []
        for c in classes:
            st = styles[c]
            class_handles.append(
                Line2D([0],[0], marker=st['marker'], linestyle='None',
                       markerfacecolor='none', markeredgecolor='k',
                       markersize=8, label=f'class {c}')
            )
        ax2.legend(handles=class_handles, title="Class (marker)", fontsize=9, loc='best')

        fig.suptitle(f'Shapley Comparison: {os.path.basename(train_csv)}', fontsize=15, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(graph_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(out_png, dpi=300); plt.close()
        print('Saved to:', out_png)

def diagnose_test_sets(gnn, base_test_dir: str, holdout_bases, out_dir: str, a = 0.05, b = 0.40,
                      ylim = (30, 105), random_iters = 10):
    _ensure_dir(out_dir)

    catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc = False)}
    for base in holdout_bases:
        if base not in catalog:
            print(f"WARNING: Base {base} not found in {base_test_dir}.")
            continue
        paths = catalog[base]

        # dlist = load_and_preprocess(paths["train_like"], paths["test"], d = gnn.skip_lin.in_features)
        ## Inject LOO
        dlist = load_and_preprocess(paths['train_like'], paths['test'], d=svd_d)
        preds = predict(gnn, dlist)

        df_tr_like = pd.read_csv(paths['train_like'])
        df_tr_like['Pred_Shap'] = preds
        out_pred_csv = os.path.join(out_dir, f'{base}_Pred.csv')
        df_tr_like.to_csv(out_pred_csv, index = False)
        print('Saved to:', out_pred_csv)

        df_test = pd.read_csv(paths['test'])
        X_train = df_tr_like.drop(columns=['target','PerDShap','Pred_Shap']).astype(float)
        # X_train = df_tr_like.drop(columns=['target', 'TrueDShap', 'Pred_Shap']).astype(float)
        X_test  = df_test.drop(columns=['target']).astype(float)
        y_train = df_tr_like['target'].to_numpy().astype(int)
        y_test  = df_test['target'].to_numpy().astype(int)

        if paths.get('tmc') is None:
            print(f"WARNING: No TMC file for {base}.")
            continue

        df_tmc   = pd.read_csv(paths['tmc'])
        TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
        Per_DShap = df_tr_like['PerDShap'].to_numpy()
        # Per_DShap = df_tr_like['TrueDShap'].to_numpy()
        Pred_Shap = preds

        # Removal plots
        desc_png = os.path.join(out_dir, f'{base}_Removal_High.png')
        _ = PerformancePlot_Removal(vals = {'Permuted Data Shapley': Per_DShap, 'TMC Data Shapley': TMC_Shap, 'Predicted Shapley': Pred_Shap},
                                    X_train=X_train.to_numpy(), y_train=y_train, X_test=X_test.to_numpy(), y_test=y_test,
                                    a = a, b = b, ylim = ylim, order = 'descending', random_iters = random_iters, save_path=desc_png)
        print('Saved to:', desc_png)

        asc_png = os.path.join(out_dir, f'{base}_Removal_Low.png')
        _ = PerformancePlot_Removal(vals = {'Permuted Data Shapley': Per_DShap, 'TMC Data Shapley': TMC_Shap, 'Predicted Shapley': Pred_Shap},
                                    X_train=X_train.to_numpy(), y_train=y_train, X_test=X_test.to_numpy(), y_test=y_test,
                                    a = a, b = b, ylim = ylim, order='ascending', random_iters=random_iters, save_path = asc_png)
        print('Saved to', asc_png)

        # Bar comparison
        tmc_vals  = np.asarray(TMC_Shap)
        per_vals  = np.asarray(Per_DShap)
        pred_vals = np.asarray(Pred_Shap)
        mean_tmc, mean_per, mean_pred = tmc_vals.mean(), per_vals.mean(), pred_vals.mean()
        N = len(per_vals); idx = np.arange(N); width = 0.27
        bins = 50
        plt.figure(figsize=(11, 8))
        plt.hist(per_vals, bins=bins, alpha=0.45, label='Permutated-based Data Shapley',
                 edgecolor='k', linewidth=0.3, color=COLORS["true"])
        plt.hist(tmc_vals, bins=bins, alpha=0.45, label='TMC Data Shapley',
                 edgecolor='k', linewidth=0.3, color=COLORS["tmc"])
        plt.hist(pred_vals, bins=bins, alpha=0.45, label='Predicted Data Shapley',
                 edgecolor='k', linewidth=0.3, color=COLORS["pred"])

        plt.axvline(mean_per, linestyle='--', linewidth=2, color=COLORS["true"], label=f'Avg Permuted ({mean_per:.3f})')
        plt.axvline(mean_tmc, linestyle='--', linewidth=2, color=COLORS["tmc"], label=f'Avg TMC ({mean_tmc:.3f})')
        plt.axvline(mean_pred, linestyle='--', linewidth=2, color=COLORS["pred"], label=f'Avg Pred ({mean_pred:.3f})')

        plt.xlabel('Shapley Value', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)
        plt.title(f'Distribution of Shapley Values — {base}', fontsize = 11)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend(fontsize=8);
        plt.tight_layout()

        bar_png = os.path.join(out_dir, f'{base}_DShapHistogram.png')
        plt.savefig(bar_png, dpi=500);
        plt.close()
        print('saved ➜', bar_png)

        # Linear comparison
        lo = min(per_vals.min(), tmc_vals.min(), pred_vals.min())
        hi = max(per_vals.max(), tmc_vals.max(), pred_vals.max())

        classes = np.unique(y_train)
        styles = _class_styles(classes)

        fig, ax = plt.subplots(figsize=(8.8, 8.8))
        for c in classes:
            mask = (y_train == c)
            st = styles[c].copy()
            ax.scatter(per_vals[mask], pred_vals[mask], **st, facecolors='none', edgecolors=COLORS["pred"], linewidths=1.2)
            ax.scatter(per_vals[mask], tmc_vals[mask],  **st, facecolors='none', edgecolors=COLORS["tmc"],  linewidths=1.2)

        ax.plot([lo - 0.001, hi + 0.001], [lo - 0.001, hi + 0.001], '--k', linewidth=1.5)
        ax.set_xlabel('Permuted Shapley Value'); ax.set_ylabel('Other Shapley Values')
        ax.set_title(f'Linear Comparison — {base}')
        ax.grid(True, linestyle='--', alpha=0.5)

        series_handles = [
            Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["pred"], markersize=8, label='Predicted Shapley'),
            Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["tmc"],  markersize=8, label='TMC Shapley'),
        ]
        legend_series = ax.legend(handles=series_handles, title="Series (color)", loc='upper left', fontsize=9)
        ax.add_artist(legend_series)

        class_handles = []
        for c in classes:
            st = styles[c]
            class_handles.append(
                Line2D([0],[0], marker=st['marker'], linestyle='None',
                       markerfacecolor='none', markeredgecolor='k', markersize=8, label=f'class {c}')
            )
        legend_classes = ax.legend(handles=class_handles, title="Class (marker)", loc='lower right', fontsize=9)
        ax.add_artist(legend_classes)

        plt.tight_layout()
        line_png = os.path.join(out_dir, f'{base}_LinearCompare.png')
        plt.savefig(line_png, dpi=500); plt.close()
        print('Saved to:', line_png)

def train_then_diagnose(BASE_TRAIN: str, BASE_TEST: str, OUT_DIR: str, holdout_bases,
                        in_channels = svd_d, hidden_channels = 256, heads = 8,
                        epochs = 500, lr = 1e-4, weight_decay = 1e-7,
                        a = 0.05, b = 0.40, ylim = (40, 105), random_iters = 10,
                        scheduler_type="step", step_size=50, step_gamma=0.5, lambda_fn=None):
    _ensure_dir(OUT_DIR)

    triplets = discover_datasets(BASE_TRAIN, require_tmc = False)
    dataset_graphs = []
    train_graph_build_sec_by_dataset = {}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0_train_graphs = time.perf_counter()

    for it in triplets:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_dataset = time.perf_counter()

        dataset_graphs.append(load_and_preprocess(it["train_like"], it["test"], d = svd_d))

        if device.type == "cuda":
            torch.cuda.synchronize()
        dataset_sec = time.perf_counter() - t0_dataset
        train_graph_build_sec_by_dataset[it["base_name"]] = float(dataset_sec)
        print(f"[train cached] {it['base_name']}: build_sec = {dataset_sec: .6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_graph_build_sec = time.perf_counter() - t0_train_graphs

    in_channels = dataset_graphs[0][0].x.size(1)
    gnn = GINShap(in_channels = in_channels, hidden_channels = hidden_channels, heads = heads)
    _ = train(gnn, dataset_graphs, epochs=epochs, lr=lr, weight_decay=weight_decay,
              base_test_dir=BASE_TEST, holdout_bases=holdout_bases, out_dir=OUT_DIR,
              scheduler_type=scheduler_type, step_size=step_size, step_gamma=step_gamma, lambda_fn=lambda_fn,
              train_graph_build_sec=train_graph_build_sec,
              train_graph_build_sec_by_dataset=train_graph_build_sec_by_dataset)

    model_path = os.path.join(OUT_DIR, "GenSHAP_S_model.pt")
    torch.save({
        "model_state_dict": gnn.state_dict(),
        "model_class": "GINShap",
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "heads": heads,
        "svd_d": svd_d,
        "extra_dim": EXTRA_DIM,
        "c_list": C_List,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler_type": scheduler_type,
        "step_size": step_size,
        "step_gamma": step_gamma,
        "train_graph_build_sec_including_sampling_marginal_probes": train_graph_build_sec,
        "train_graph_build_sec_by_dataset": train_graph_build_sec_by_dataset,
    }, model_path)
    print("saved ➜", model_path)

    train_dir = os.path.join(OUT_DIR, "diagnostics_train_final"); _ensure_dir(train_dir)
    diagnose_train_sets(gnn, base_train_dir=BASE_TRAIN, out_dir=train_dir, graph_dir=train_dir)

    test_dir  = os.path.join(OUT_DIR, "diagnostics_test_final");  _ensure_dir(test_dir)
    diagnose_test_sets(gnn, base_test_dir=BASE_TEST, holdout_bases=holdout_bases,
                       out_dir=test_dir, a=a, b=b, ylim=ylim, random_iters=random_iters)
    return gnn

#####################################################################################################
#####################################################################################################
########################################  Experiments ###############################################

if __name__ == "__main__":
    BASE_TRAIN = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Base_LR/Random_PCA/Train'
    BASE_TEST  = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Base_LR/Random_PCA/Test'
    OUT_DIR    = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260415/Trial05'
    HOLDOUTS   = ['Iris_1', 'Vehicle_01_1', 'characters_048_1', 'STL10_3568x_3', 'volkert_15_2', 'volkert_1245_1']

    _ = train_then_diagnose(
        BASE_TRAIN, BASE_TEST, OUT_DIR, HOLDOUTS,
        in_channels=FEAT_DIM, hidden_channels=64, heads=2,
        epochs=1000, lr=1e-5, weight_decay=5e-7,
        scheduler_type="None"
        # , step_size=50, step_gamma = 0.5
    )
