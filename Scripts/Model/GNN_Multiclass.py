'''
0. Import Packages
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
FEAT_DIM = svd_d + 9

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
C_List = [0.25, 0.5, 1, 2, 4, 8]

def build_graph(X_raw, y_raw, phi, total_util, c, loo_z, grad_norm_z, grad_err_z, d=svd_d):
    D = pairwise_distances(X_raw, metric='euclidean')
    ## Gaussian Kernel
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))

    ## Injection of statistics(mean, var, min, max, skewness, kurtosis) of each row
    def _standardize(arr):
        mu = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr - mu) / std

    def _skewness_1d(v, eps=1e-12):
        v = np.asarray(v)
        m = v.mean()
        c = v - m
        s2 = np.mean(c**2)
        s = np.sqrt(s2 + eps)
        return np.mean(c**3) / (s**3 + eps)

    def _kurtosis_1d(v, eps=1e-12):
        v = np.asarray(v, dtype=float)
        m = v.mean()
        c = v - m
        s2 = np.mean(c ** 2)
        s4 = (s2 + eps) ** 2
        return np.mean(c ** 4) / s4 - 3.0

    Xr = np.asarray(X_raw, dtype=float)
    row_mean = Xr.mean(axis=1)
    row_var = Xr.var(axis=1)
    row_min = Xr.min(axis=1)
    row_max = Xr.max(axis=1)
    row_skew = np.array([_skewness_1d(Xr[i]) for i in range(Xr.shape[0])])
    row_kurt = np.array([_kurtosis_1d(Xr[i]) for i in range(Xr.shape[0])])

    stats = np.vstack([row_mean, row_var, row_min, row_max, row_skew, row_kurt]).T  # (N, 6)
    stats_z = np.column_stack([_standardize(stats[:, j]) for j in range(stats.shape[1])])

    X_svd = SVD_Transform(A, d = d)
    y = np.asarray(y_raw).ravel()
    A_y = (y[:, None] == y[None, :]).astype(float)


    ## Injection of LOO + Gradient Norm + Grad Error
    loo_z = np.asarray(loo_z).reshape(-1, 1)  # (N, 1)
    grad_norm_z = np.asarray(grad_norm_z).reshape(-1, 1)  # (N, 1)
    grad_err_z = np.asarray(grad_err_z).reshape(-1, 1)  # (N, 1)

    ## weight on injected features
    weight_extra = 3.0
    extra_feats = np.concatenate([loo_z, grad_norm_z, grad_err_z, stats_z], axis=1)  # (N, 9)
    extra_feats *= weight_extra
    X_node = np.concatenate([X_svd, extra_feats], axis=1)


    N = X_raw.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(np.concatenate([eidx, eidx[::-1]], axis=1), dtype=torch.long, device=device)

    w_feat = A[src, dst]
    w_label = A_y[src, dst]
    edge_attr_feat = torch.tensor(np.concatenate([w_feat, w_feat]), dtype=torch.float, device=device).unsqueeze(1)
    edge_attr_label = torch.tensor(np.concatenate([w_label, w_label]), dtype=torch.float, device=device).unsqueeze(1)

    data = Data(
        x=torch.tensor(X_node, dtype=torch.float, device=device),
        edge_index=edge_index,
        edge_attr_feat=edge_attr_feat,
        edge_attr_label=edge_attr_label
    )

    # Extra attributes
    data.X_raw = torch.tensor(X_raw, dtype=torch.float, device=device)
    data.phi = torch.tensor(phi, dtype=torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype=torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype=torch.long, device=device)
    data.num_classes = int(len(np.unique(y)))

    # Record number of classes of each dataset
    K = int(np.unique(y).size)
    k_vec = np.array([K, 1.0 / max(1, K), np.log1p(K)], dtype=float)
    data.k_feat = torch.tensor(np.tile(k_vec, (N, 1)), dtype=torch.float, device=device)

    ## Keep LOO_z, grad_norm_z, grad_err_z for diagnostics
    data.loo_z = torch.tensor(loo_z, dtype=torch.float, device=device)
    data.grad_norm_z = torch.tensor(grad_norm_z, dtype=torch.float, device=device)
    data.grad_err_z  = torch.tensor(grad_err_z, dtype=torch.float, device=device)
    data.stats_z = torch.tensor(stats_z, dtype=torch.float, device=device)
    return data


'''
3. Data Preprocessing
'''
def load_and_preprocess(train_csv, test_csv, d = svd_d):
    df_train = pd.read_csv(train_csv)
    X_raw = df_train.drop(columns=['target','PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy()
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy().astype(int)

    lr_full = LogisticRegression(max_iter = 100, random_state=23)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    ## Compute gradient norm and gradient error
    n, d_feat = X_raw.shape
    P = lr_full.predict_proba(X_raw)  # (N, C)
    classes = lr_full.classes_  # shape (C,)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    n = X_raw.shape[0]
    col_idx = np.array([class_to_idx[y] for y in y_raw])  # (N,)

    Y_onehot = np.zeros_like(P)
    Y_onehot[np.arange(n), col_idx] = 1.0

    grad_w = (P - Y_onehot)[:, :, None] * X_raw[:, None, :]  # (N, C, d_feat)
    grad_norm = np.linalg.norm(grad_w, axis=(1, 2))  # (N,)

    p_true = P[np.arange(n), col_idx]  # (N,)
    grad_err = 1.0 - p_true
    def _standardize(arr):
        mu = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr - mu) / std

    grad_norm_z = _standardize(grad_norm)
    grad_err_z  = _standardize(grad_err)

    ## Compute LOO values
    n = X_raw.shape[0]
    loo_raw = np.zeros(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        lr_i = LogisticRegression(max_iter=100, random_state=23)
        lr_i.fit(X_raw[mask], y_raw[mask])
        acc_i = lr_i.score(X_test, y_test)

        loo_raw[i] = model_acc - acc_i  # LOO_i = U(D) - U(D\{i})

    mu = loo_raw.mean()
    std = loo_raw.std()
    if std < 1e-12:
        loo_z = np.zeros_like(loo_raw)
    else:
        loo_z = (loo_raw - mu) / std

    data_list = []
    for c in C_List:
        data = build_graph(X_raw, y_raw, phi, total_util, c=c, loo_z=loo_z, grad_norm_z=grad_norm_z,
                             grad_err_z=grad_err_z, d=d)
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
    def __init__(self, in_channels, hidden_channels = 64, out_channels = 1, heads = 2):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))

        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(2):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                            aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
                            activation = ReLU())
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))
        d_concat = hidden_channels * len(C_List)

        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
        self.gt = TransformerEncoder(encoder_layer, num_layers = 2)

        self.pre_gt = Linear(in_channels, d_concat)

        self.Z_gin = Sequential(Linear(d_concat, hidden_channels), ReLU())
        self.Z_gt = Sequential(Linear(d_concat, hidden_channels), ReLU())

        self.out = Sequential(
            Linear( (hidden_channels * 2) + 3, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, 100),
            ReLU(),
            Linear(100, out_channels)
        )

    def forward(self, data_list):
        data = data_list[0].to(device)
        x0 = data.x
        output_GIN = []
        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x.to(device)
            edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label

            ## GPU Version
            g = dgl.graph(
                (gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
                num_nodes=node_input.size(0),device=device
            )

            ## CPU Version
            # g = dgl.graph(
            #     (gdata.edge_index[0], gdata.edge_index[1]),
            #     num_nodes=node_input.size(0)
            # )
            g.edata['w'] = edge_weight

            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])
            output_GIN.append(node_input)

        gin_cat = torch.cat(output_GIN, dim = 1)
        gin_cat = gin_cat + self.skip_lin(x0)

        h_gin = self.Z_gin(gin_cat)

        gt_in = self.pre_gt(x0)
        gt_in = gt_in.unsqueeze(0)
        gt_out = self.gt(gt_in).squeeze(0)
        h_gt = self.Z_gt(gt_out)

        h_fused = torch.cat([h_gin, h_gt], dim=1)

        k_col = data.k_feat
        h_plus = torch.cat([h_fused, k_col], dim = 1)
        return self.out(h_plus)


'''
5. Training (standard scheme)
'''
def train(model, dataset_graphs, epochs=500, lr=1e-5, weight_decay=1e-7, base_test_dir: str = None,
          holdout_bases: list = None, out_dir: str = None, scheduler_type: str = "step", step_size: int = 50,
          step_gamma: float = 0.5, lambda_fn = None, log_every: int = 5):

    model.to(device)

    ## Record time
    train_time_hist = []
    test_time_hist = []

    if out_dir:
        os.makedirs(out_dir, exist_ok = True)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)

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

    loss_history = []
    per_epoch_rows = []

    catalog = None
    if holdout_bases and base_test_dir:
        catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc=False)}

    for ep in range(1, epochs + 1):
        if device.type == "cuda": torch.cuda.synchronize()
        t0_train = time.perf_counter()
        'TRAIN'
        model.train()
        losses = []
        for data_list in dataset_graphs:
            total_util = data_list[0].total_util.to(device)
            phi_true = data_list[0].phi.to(device).view(-1)
            N = phi_true.numel()
            scale = N/total_util
            phi_true_scaled = phi_true * scale
            phi_hat_scaled = model(data_list).view(-1)
            loss = F.mse_loss(phi_true_scaled, phi_hat_scaled, reduction = "mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        if device.type == "cuda": torch.cuda.synchronize()
        train_time_hist.append(time.perf_counter() - t0_train)

        if scheduler is not None:
            scheduler.step()

        L = torch.stack(losses).mean()
        loss_history.append(L.item())

        avg_mse = None
        if catalog is not None and (holdout_bases is not None) and len(holdout_bases) > 0:
            if device.type == "cuda": torch.cuda.synchronize()
            t0_test = time.perf_counter()

            model.eval()
            row = {"epoch":ep}
            mses = []

            for base in holdout_bases:
                p = catalog.get(base, None)
                if p is None:
                    continue

                dl = load_and_preprocess(p["train_like"], p["test"], d = svd_d)

                with torch.no_grad():
                    pred_scaled = model(dl).view(-1)
                    phi_t = dl[0].phi.to(device).view(-1)
                    total_util_t = dl[0].total_util.to(device)
                    N_t = phi_t.numel()
                    phi_t_scaled = phi_t * (N_t / total_util_t)
                    m = F.mse_loss(pred_scaled, phi_t_scaled, reduction="mean").item()

                row[base] = m
                mses.append(m)

            if mses:
                avg_mse = float(np.mean(mses))
                row["avg_mse"] = avg_mse
                per_epoch_rows.append(row)

            if device.type == "cuda": torch.cuda.synchronize()
            test_time_hist.append(time.perf_counter() - t0_test)
        else:
            test_time_hist.append(0.0)

        if (ep % log_every == 0) or (ep == 1) or (ep == epochs):
            if avg_mse is None:
                print(f"Epoch {ep:5d} | Train MSE: {L.item():.8e}")
            else:
                print(f"Epoch {ep:5d} | Train MSE: {L.item():.8e} | Test MSE: {avg_mse:.8e}")

    if out_dir:
        xs = np.arange(len(loss_history))
        ys = np.array(loss_history)
        ## Save training mse
        pd.DataFrame({"epoch": xs + 1, "train_mse": ys}).to_csv(os.path.join(out_dir, "Training_MSE.csv"), index = False)
        ## Save train & test time
        time_txt = os.path.join(out_dir, "Train_Test_Time.txt")
        with open(time_txt, "w") as f:
            f.write("epoch,train_sec,test_sec\n")
            for ep, tr, te in zip(xs + 1, train_time_hist, test_time_hist):
                f.write(f"{ep},{float(tr):.6f},{float(te):.6f}\n")

        if per_epoch_rows:
            df_full = pd.DataFrame(per_epoch_rows)
            csv_full = os.path.join(out_dir, "Holdout_MSE_Full.csv")
            df_full.to_csv(csv_full, index = False); print("saved to:", csv_full)

            df_train_full = pd.DataFrame({"epoch": xs + 1, "train_mse": ys})
            df_tt = df_full.merge(df_train_full, on = "epoch", how = "left")
            df_tt = df_tt[df_tt["epoch"] >= 20]

            plt.figure(figsize = (6, 4))
            plt.plot(df_tt["epoch"], df_tt["avg_mse"], label = "Test(avg MSE)", linewidth = 1.2, alpha = 0.9)
            plt.plot(df_tt["epoch"], df_tt["train_mse"], label = "Train (MSE)", linewidth = 1.2, alpha = 0.9)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title('Optimisation Curve of Training & Testing Sets (MSE)')
            plt.grid(True, ls = '--', alpha = 0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "Optimisation_Curve.png"))
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(xs, np.log(ys + 1e-12), label='Training Loss (log)', linewidth=1.2, alpha=0.9)
            plt.xlabel('Epochs'); plt.ylabel('log(MSE)'); plt.title('Optimisation Curve(log)')
            plt.legend(); plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'Optimisation_Curve_log.png')); plt.close()

    return loss_history


'''
6. Prediction
'''
def predict(model, data_list):
    model.eval()
    with torch.no_grad():
        yhat_scaled = model(data_list).view(-1)
        total_util = data_list[0].total_util.to(yhat_scaled.device)
        N = data_list[0].x.size(0)
        descaled = yhat_scaled * (total_util / N)
        return descaled.cpu().numpy()


'''
7. Supplementary Functions
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
    dataset_graphs = [load_and_preprocess(it["train_like"], it["test"], d = svd_d) for it in triplets]

    gnn = GINShap(in_channels = in_channels, hidden_channels = hidden_channels, heads = heads)
    _ = train(gnn, dataset_graphs, epochs=epochs, lr=lr,
              base_test_dir=BASE_TEST, holdout_bases=holdout_bases, out_dir=OUT_DIR,
              scheduler_type=scheduler_type, step_size=step_size, step_gamma=step_gamma, lambda_fn=lambda_fn)

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
    BASE_TRAIN = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Train'
    BASE_TEST  = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test'
    OUT_DIR    = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Outputs_II/260123/Trial0'
    HOLDOUTS   = ['Iris_1', 'Vehicle_01_1', 'characters_048_1']

    _ = train_then_diagnose(
        BASE_TRAIN, BASE_TEST, OUT_DIR, HOLDOUTS,
        in_channels=FEAT_DIM, hidden_channels=64, heads=2,
        epochs=2, lr=1e-5, weight_decay=1e-7,
        scheduler_type="None", step_size=50, step_gamma = 0.5
    )

