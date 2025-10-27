'''
0. Import Packages
'''
import os
import re
import dgl
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dgl.nn.pytorch.conv import GINConv
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR, LambdaLR, OneCycleLR
from torch.nn import (Sequential, Linear, LeakyReLU, ReLU, TransformerEncoder, TransformerEncoderLayer, LayerNorm)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal

'Check Availability of GPU'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))


'''
1. Randomized SVD
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

def SVD_Transform(A, d=100):
    # U, S, _ = randomized_svd(A, n_components=d, random_state = 23) #change to full svd/torch svd
    # X_svd = U * np.sqrt(S)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    X_svd = U[:, :d] * np.sqrt(S[:d])
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = flip_pe_sign(X_t, method='sum')  # l0 (positive) / sum
    return X_t.numpy()


'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4, 8]

def build_graph(X_raw, y_raw, phi, total_util, c, d = 100):
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    X_svd = SVD_Transform(A, d = d)
    y = np.asarray(y_raw).ravel() # Ensure 1D array
    A_y = (y[:, None] == y[None, :]).astype(float) # Label similarity matrix

    N = X_raw.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(np.concatenate([eidx, eidx[::-1]], axis = 1), dtype = torch.long, device=device)

    # Extract upper-triangle pairs
    w_feat = A[src, dst]
    w_label = A_y[src, dst]
    edge_attr_feat = torch.tensor(np.concatenate([w_feat, w_feat]), dtype = torch.float, device=device).unsqueeze(1)
    edge_attr_label = torch.tensor(np.concatenate([w_label, w_label]), dtype = torch.float, device=device).unsqueeze(1)

    ## Scaling phi
    scale = float(N)/float(total_util) if total_util is not None else 1.0
    phi_scaled = phi * scale

    data = Data(
        x = torch.tensor(X_svd, dtype = torch.float, device=device),
        edge_index = edge_index,
        edge_attr_feat = edge_attr_feat,
        edge_attr_label = edge_attr_label
    )
    # Extra attributes
    data.X_raw = torch.tensor(X_raw, dtype = torch.float, device=device)
    data.phi = torch.tensor(phi_scaled, dtype = torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype = torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype = torch.long, device=device)

    return data


'''
3. Data Preprocessing
'''
def load_and_preprocess(train_csv, test_csv, d=100):
    df_train = pd.read_csv(train_csv)
    X_raw = df_train.drop(columns=['target','PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy()
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy()

    lr_full = LogisticRegression(max_iter=100, random_state=23)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    data_list = [] # length = 5
    for c in C_List:
        data = build_graph(X_raw, y_raw, phi, total_util, d = d, c = c)
        data_list.append(data)

    return data_list


'''
4. Model Construction
'''
'Version A: With LayerNorm & 5 separate conv stacks'
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels = 128, out_channels = 1, heads = 4):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))

        # 5 Parallel GIN Branches
        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(3):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                            aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
                            activation = ReLU())
                )
            self.branches.append(convs)  # length = 6

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))

        # GT (via TransformerEncoder)
        d_concat = hidden_channels * len(C_List)
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
        self.gt = TransformerEncoder(encoder_layer, num_layers = 3)

        # Z
        self.Z = Sequential(Linear(d_concat, hidden_channels),ReLU())

        # MLP (widen by hidden_channels * 2)
        self.out = Sequential(
            Linear(hidden_channels + 2, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, 100),
            ReLU(),
            Linear(100, out_channels)
        )

    def forward(self, data_list):
        data = data_list[0].to(device)
        x0 = data.x # Extract nodes
        ei = data.edge_index # Extract edge indices

        # DGL Graph
        g = dgl.graph((ei[0], ei[1]), num_nodes = x0.size(0))

        # Parallel GIN
        output_GIN = []

        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x.to(device)
            edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label

            g = dgl.graph((gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
                            num_nodes=node_input.size(0), device=device)
            g.edata['w'] = edge_weight

            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])
            output_GIN.append(node_input)

        # Skip-Connection
        output_GIN = torch.cat(output_GIN, dim = 1)
        output_skip = output_GIN + self.skip_lin(x0)

        # GT
        output_skip = output_skip.unsqueeze(0)
        output_gt = self.gt(output_skip)
        output_gt = output_gt.squeeze(0)

        # H
        output_h = self.Z(output_gt)

        # total utility + (N/10) -> MLP
        N = output_h.size(0)
        total_util = data.total_util
        utility_vec = total_util.expand(N, 1)
        NU_vector = torch.full((N, 1), np.sqrt(N)/10., device = device)
        output_final = torch.cat([output_h, utility_vec, NU_vector], dim = 1)

        return self.out(output_final)


# 'Version B: With LayerNorm & 1 shared conv stack'
# class GINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels = 256, out_channels = 1, heads = 8):
#         super().__init__()
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LayerNorm(o), LeakyReLU(), Linear(o, o))
#
#         # 5 Parallel GIN Branches
#         self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.shared_convs = torch.nn.ModuleList([
#             GINConv(mlp(in_channels, hidden_channels),
#                     aggregator_type='sum', init_eps=0.1, learn_eps=True,
#                     activation=LeakyReLU()),
#             GINConv(mlp(hidden_channels, hidden_channels),
#                     aggregator_type='sum', init_eps=0.1, learn_eps=True,
#                     activation=LeakyReLU()),
#             GINConv(mlp(hidden_channels, hidden_channels),
#                     aggregator_type='sum', init_eps=0.1, learn_eps=True,
#                     activation=LeakyReLU()),
#         ])
#
#         self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))
#
#         # GT (via TransformerEncoder)
#         d_concat = hidden_channels * len(C_List)
#         encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
#         self.gt = TransformerEncoder(encoder_layer, num_layers = 3)
#
#         # Z
#         self.Z = Sequential(Linear(d_concat, hidden_channels),LeakyReLU())
#
#         # MLP (widen by hidden_channels * 2)
#         self.out = Sequential(
#             Linear(hidden_channels + 2, hidden_channels * 2),
#             ReLU(),
#             Linear(hidden_channels * 2, 100),
#             ReLU(),
#             Linear(100, out_channels)
#         )
#
#     def forward(self, data_list):
#         data = data_list[0].to(device)
#         x0 = data.x # Extract nodes
#         ei = data.edge_index # Extract edge indices
#
#         # DGL Graph
#         g = dgl.graph((ei[0], ei[1]), num_nodes = x0.size(0))
#
#         # Parallel GIN
#         output_GIN = []
#
#         for idx, gdata in enumerate(data_list):
#             node_input = gdata.x.to(device)
#             edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)  # (E,)
#             edge_attr_label = gdata.edge_attr_label.view(-1).to(device)  # (E,)
#
#             weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
#             edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label  # (E,)
#
#             g = dgl.graph((gdata.edge_index[0], gdata.edge_index[1]),
#                           num_nodes=node_input.size(0), device=device)
#             g.edata['w'] = edge_weight
#             for conv in self.shared_convs:
#                 node_input = conv(g, node_input, edge_weight=g.edata['w'])
#             output_GIN.append(node_input)
#
#         # Skip-Connection
#         output_GIN = torch.cat(output_GIN, dim = 1)
#         output_skip = output_GIN + self.skip_lin(x0)
#
#         # GT
#         output_skip = output_skip.unsqueeze(0) # transform to: (1, N, hidden_channels*5)
#         output_gt = self.gt(output_skip)
#         output_gt = output_gt.squeeze(0)
#
#         # H
#         output_h = self.Z(output_gt)
#
#         # total utility + (N/10) -> MLP
#         N = output_h.size(0)
#         total_util = data.total_util
#         utility_vec = total_util.expand(N, 1)
#         NU_vector = torch.full((N, 1), np.sqrt(N)/10., device = device)
#         output_final = torch.cat([output_h, utility_vec, NU_vector], dim = 1)
#
#         return self.out(output_final)


'''
5. Training
'''
# def listnet_loss(phi_hat, phi_true, tau = 0.01):
#     log_p_hat = F.log_softmax(phi_hat/tau, dim = 0)
#     p_true = F.softmax(phi_true/tau, dim = 0)
#     return -(p_true * log_p_hat).sum()

# def ranknet_loss(phi_hat, phi_true):
#     N = phi_hat.size(0)
#     idx_i, idx_j = torch.triu_indices(N, N, offset = 1)
#     S = torch.sign(phi_true[idx_i] - phi_true[idx_j])
#     diffs = phi_hat[idx_i] - phi_hat[idx_j]
#     return F.softplus(-S * diffs).mean()

# def pairwise_rank_loss(phi_hat, phi_true, margin = 0):
#     N = phi_hat.size(0)
#     idx_i, idx_j = torch.triu_indices(N, N, offset = 1)
#     y = torch.sign(phi_true[idx_i] - phi_true[idx_j])
#     return F.margin_ranking_loss(phi_hat[idx_i], phi_hat[idx_j], y, margin = margin)

# def BPR_loss(phi_hat, phi_true):
#     N = phi_true.size(0)
#     idx_i, idx_j = torch.triu_indices(N, N, offset = 1)
#     mask = (phi_true[idx_i] > phi_true[idx_j])
#     pos_i = idx_i[mask]
#     neg_j = idx_j[mask]
#     return -F.logsigmoid(phi_hat[pos_i] - phi_hat[neg_j]).mean()
# ' Version 0. Constant LR'
# def train(model, dataset_graphs, epochs=1000, lr=3e-5, weight_decay = 1e-7,
#           base_test_dir: str = None, holdout_bases: list = None, out_dir: str = None,
#           ckpt_every: int = 10, checkpoint_dir: str = None,
#           use_onecycle: bool = False, mx_lr: float = None, onecycle_kwargs: dict = None):
#     model.to(device)
#
#     if out_dir: os.makedirs(out_dir, exist_ok = True)
#     if checkpoint_dir is None and out_dir:
#         checkpoint_dir = os.path.join(out_dir, 'checkpoints')
#     if checkpoint_dir: os.makedirs(checkpoint_dir, exist_ok = True)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay) # adamw
#
#     loss_history = []
#     per_epoch_rows = []
#     sparse_ckpt_rows = []
#
#     for ep in range(1, epochs + 1):
#         model.train()
#         losses = []
#
#         for data_list in dataset_graphs:
#             phi_hat = model(data_list).view(-1)
#             phi_true = data_list[0].phi.to(device)
#             loss_MSE = F.mse_loss(phi_hat, phi_true, reduction='mean')
#
#             loss = loss_MSE
#             losses.append(loss)
#         L = torch.stack(losses).mean()
#         loss_history.append(L.item())
#
#         optimizer.zero_grad()
#         L.backward()
#         optimizer.step()
#         # scheduler.step()
#
#         if ep % 5 == 0:
#             # current_lr = scheduler.get_last_lr()[0]
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f'Epoch {ep:3d} | total loss {L:.16f}| LR: {current_lr:.2e}')
#
#         avg_mse = None
#         if holdout_bases and base_test_dir:
#             model.eval()
#             row = {"epoch": ep}
#             with torch.no_grad():
#                 mses = []
#                 for base in holdout_bases:
#                     p = _mk_paths(base_test_dir, base)
#                     dl = load_and_preprocess(p['train_like'], p['test'])
#                     pred = model(dl).view(-1)
#                     phi_t = dl[0].phi.to(device).view(-1)
#                     m = F.mse_loss(pred, phi_t, reduction='mean').item()
#                     row[base] = m
#                     mses.append(m)
#                 if mses:
#                     avg_mse = float(np.mean(mses))
#                     row["avg_mse"] = avg_mse
#                     per_epoch_rows.append(row)
#                     print(f'>>> Test avg MSE @ {ep}: {avg_mse:.6f}')
#
#
#         if ckpt_every and (ep % ckpt_every == 0) and checkpoint_dir:
#             path = os.path.join(checkpoint_dir, f"model_ep{ep}.pt")
#             torch.save({
#                 "epoch": ep,
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "loss_train_mse": L.item(),
#                 "avg_test_mse": avg_mse,
#             }, path)
#             print("saved checkpoint ➜", path)
#             if avg_mse is not None:
#                 sparse_ckpt_rows.append({"epoch": ep, "avg_mse": avg_mse})
#
#     if checkpoint_dir:
#         final_path = os.path.join(checkpoint_dir, f"model_final_ep{epochs}.pt")
#         torch.save({
#             "epoch": epochs,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "loss_train_mse": loss_history[-1],
#             "avg_test_mse": (per_epoch_rows[-1]["avg_mse"] if per_epoch_rows else None),
#         }, final_path)
#         print("saved final model ➜", final_path)
#
#     if out_dir:
#         # Training MSE curve
#         xs = np.arange(len(loss_history)); ys = np.array(loss_history)
#         # [CHANGED] also save full train-MSE history to CSV (epoch, train_mse)
#         pd.DataFrame({"epoch": xs + 1, "train_mse": ys}).to_csv(
#             os.path.join(out_dir, "Training_MSE_full.csv"), index=False
#         )
#
#         plt.figure(figsize=(6,4)); plt.plot(xs, ys); plt.grid(True, ls='--', alpha=0.5)
#         plt.xlabel("Epoch"); plt.ylabel("Train MSE"); plt.title("Training Curve")
#         plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Training_MSE.png")); plt.close()
#
#         # Test MSE (every epoch)
#         if per_epoch_rows:
#             df_full = pd.DataFrame(per_epoch_rows)
#             csv_full = os.path.join(out_dir, "Holdout_MSE_full.csv")
#             df_full.to_csv(csv_full, index=False); print("saved ➜", csv_full)
#
#             df_train_full = pd.DataFrame({"epoch": xs + 1, "train_mse": ys})
#             df_tt = df_full.merge(df_train_full, on="epoch", how="left")
#             df_tt = df_tt[df_tt["epoch"] >= 20]
#
#             plt.figure(figsize=(6, 4))
#             plt.plot(df_tt["epoch"], df_tt["avg_mse"], label="Test (avg MSE)",
#                      linewidth=1.2, alpha=0.9)
#             plt.plot(df_tt["epoch"], df_tt["train_mse"], label="Train (MSE)",
#                      linewidth=1.2, alpha=0.9)
#             plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Train vs Test MSE (every epoch)")
#             plt.grid(True, ls='--', alpha=0.5); plt.legend()
#             plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Train_vs_Test_MSE_every_epoch.png")); plt.close()
#
#             best_i = int(df_full['avg_mse'].idxmin()); best = df_full.iloc[best_i]
#             plt.figure(figsize=(6,4))
#             plt.plot(df_full["epoch"], df_full["avg_mse"], marker='o', lw=1.5)
#             plt.scatter([best['epoch']], [best['avg_mse']])
#             plt.text(best['epoch'], best['avg_mse'], f"  min@{int(best['epoch'])}: {best['avg_mse']:.4f}",
#                      ha='left', va='bottom', fontsize=9)
#             plt.xlabel("Epoch"); plt.ylabel("Avg Test MSE"); plt.title("Avg Test MSE (every epoch)")
#             plt.grid(True, ls='--', alpha=0.5)
#             plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Holdout_MSE_every_epoch.png")); plt.close()
#
#             # Optimisation curves (from epoch 50)
#             start_epoch = 50
#             plot_start = 0 if len(ys) <= start_epoch else start_epoch
#             xs2, ys2 = xs[plot_start:], ys[plot_start:]
#
#             plt.figure(figsize=(6, 4))
#             plt.plot(xs2, np.log(ys2 + 1e-12), label='Training Loss (log)', linewidth=1.2, alpha=0.9)  # [CHANGED]
#             plt.xlabel('Epochs'); plt.ylabel('log(MSE)'); plt.title(f'Optimisation Curve (from epoch {plot_start}, log)')
#             plt.legend(); plt.grid(True, ls='--', alpha=0.5)
#             plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'Optimisation_Curve_log.png')); plt.close()
#
#             plt.figure(figsize=(6, 4))
#             plt.plot(xs2, ys2, label='Training Loss', linewidth=1.2, alpha=0.9)  # [CHANGED]
#             plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.title(f'Optimisation Curve (from epoch {plot_start})')
#             plt.legend(); plt.grid(True, ls='--', alpha=0.5)
#             plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'Optimisation_Curve.png')); plt.close()
#
#             # sparse-at-ckpt CSV
#             if sparse_ckpt_rows:
#                 pd.DataFrame(sparse_ckpt_rows).to_csv(
#                     os.path.join(out_dir, "Holdout_MSE_sparse_at_checkpoints.csv"), index=False
#                 )
#
#     return loss_history

' Version 1. OneCycleLR '
def train(model, dataset_graphs, epochs=1000, lr=5e-5, weight_decay=1e-7,
          base_test_dir: str = None, holdout_bases: list = None, out_dir: str = None,
          ckpt_every: int = 10, checkpoint_dir: str = None,
          use_onecycle: bool = False, max_lr: float = None, onecycle_kwargs: dict = None ):

    model.to(device)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if checkpoint_dir is None and out_dir:
        checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    init_lr = lr
    if use_onecycle and max_lr is not None and onecycle_kwargs and "div_factor" in onecycle_kwargs:
        init_lr = float(max_lr) / float(onecycle_kwargs["div_factor"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    scheduler = None
    if use_onecycle:
        if max_lr is None:
            raise ValueError("use_onecycle=True requires max_lr.")
        steps_per_epoch = max(1, len(dataset_graphs))
        total_steps = epochs * steps_per_epoch

        oc_defaults = dict(
            pct_start=0.3, anneal_strategy='cos',
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
            div_factor=25.0, final_div_factor=1e4, three_phase=False)
        if onecycle_kwargs:
            oc_defaults.update(onecycle_kwargs)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            **oc_defaults
        )

    loss_history = []
    per_epoch_rows = []
    sparse_ckpt_rows = []

    def _current_lr(opt):
        return float(opt.param_groups[0]['lr'])

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for data_list in dataset_graphs:
            phi_hat  = model(data_list).view(-1)
            phi_true = data_list[0].phi.to(device)
            loss     = F.mse_loss(phi_hat, phi_true, reduction='mean')
            # loss = F.huber_loss(phi_hat, phi_true, delta = 0.01)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            losses.append(loss)

        L = torch.stack(losses).mean()
        loss_history.append(L.item())

        if ep % 5 == 0:
            print(f'Epoch {ep:3d} | total loss {L:.6e} | LR: {_current_lr(optimizer):.2e}')

        avg_mse = None
        if holdout_bases and base_test_dir:
            model.eval()
            row = {"epoch": ep}
            with torch.no_grad():
                mses = []
                # for base in holdout_bases:
                #     p = _mk_paths(base_test_dir, base)
                #     dl = load_and_preprocess(p['train_like'], p['test'])
                #     pred = model(dl).view(-1)
                #     phi_t = dl[0].phi.to(device).view(-1)
                #     # m = F.mse_loss(pred, phi_t, reduction='mean').item()
                #     m = F.huber_loss(pred, phi_t, delta = 0.01)
                #     row[base] = m
                #     mses.append(m)
                mses = []
                catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc=False)}
                for base in holdout_bases:
                    if base not in catalog:
                        print(f"WARNING: '{base}' not found in {base_test_dir}.")
                        continue
                    paths = catalog[base]
                    dl = load_and_preprocess(paths['train_like'], paths['test'])
                    pred = model(dl).view(-1)
                    phi_t = dl[0].phi.to(device).view(-1)
                    # m = F.huber_loss(pred, phi_t, delta=0.01).item()  # <- see fix #2
                    m = F.mse_loss(pred, phi_t, reduction='mean')
                    row[base] = m
                    mses.append(m)
                if mses:
                    avg_mse = float(np.mean(mses))
                    row["avg_mse"] = avg_mse
                    per_epoch_rows.append(row)
                    print(f'>>> Test avg MSE @ {ep}: {avg_mse:.6f}')

        if ckpt_every and (ep % ckpt_every == 0) and checkpoint_dir:
            path = os.path.join(checkpoint_dir, f"model_ep{ep}.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss_train_mse": L.item(),
                "avg_test_mse": avg_mse,
            }, path)
            print("saved checkpoint ➜", path)
            if avg_mse is not None:
                sparse_ckpt_rows.append({"epoch": ep, "avg_mse": avg_mse})

    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, f"model_final_ep{epochs}.pt")
        torch.save({
            "epoch": epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_train_mse": loss_history[-1],
            "avg_test_mse": (per_epoch_rows[-1]["avg_mse"] if per_epoch_rows else None),
        }, final_path)
        print("saved final model ➜", final_path)

    if out_dir:
        xs = np.arange(len(loss_history)); ys = np.array(loss_history)
        pd.DataFrame({"epoch": xs + 1, "train_mse": ys}).to_csv(
            os.path.join(out_dir, "Training_MSE_full.csv"), index=False
        )

        plt.figure(figsize=(6,4)); plt.plot(xs, ys); plt.grid(True, ls='--', alpha=0.5)
        plt.xlabel("Epoch"); plt.ylabel("Train MSE"); plt.title("Training Curve")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Training_MSE.png")); plt.close()

        if per_epoch_rows:
            df_full = pd.DataFrame(per_epoch_rows)
            csv_full = os.path.join(out_dir, "Holdout_MSE_full.csv")
            df_full.to_csv(csv_full, index=False); print("saved ➜", csv_full)

            df_train_full = pd.DataFrame({"epoch": xs + 1, "train_mse": ys})
            df_tt = df_full.merge(df_train_full, on="epoch", how="left")
            df_tt = df_tt[df_tt["epoch"] >= 20]

            plt.figure(figsize=(6, 4))
            plt.plot(df_tt["epoch"], df_tt["avg_mse"], label="Test (avg MSE)", linewidth=1.2, alpha=0.9)
            plt.plot(df_tt["epoch"], df_tt["train_mse"], label="Train (MSE)", linewidth=1.2, alpha=0.9)
            plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Train vs Test MSE (every epoch)")
            plt.grid(True, ls='--', alpha=0.5); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Train_vs_Test_MSE_every_epoch.png")); plt.close()

            best_i = int(df_full['avg_mse'].idxmin()); best = df_full.iloc[best_i]
            plt.figure(figsize=(6,4))
            plt.plot(df_full["epoch"], df_full["avg_mse"], marker='o', lw=1.5)
            plt.scatter([best['epoch']], [best['avg_mse']])
            plt.text(best['epoch'], best['avg_mse'], f"  min@{int(best['epoch'])}: {best['avg_mse']:.4f}",
                     ha='left', va='bottom', fontsize=9)
            plt.xlabel("Epoch"); plt.ylabel("Avg Test MSE"); plt.title("Avg Test MSE (every epoch)")
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Holdout_MSE_every_epoch.png")); plt.close()

            start_epoch = 50
            plot_start = 0 if len(ys) <= start_epoch else start_epoch
            xs2, ys2 = xs[plot_start:], ys[plot_start:]

            plt.figure(figsize=(6, 4))
            plt.plot(xs2, np.log(ys2 + 1e-12), label='Training Loss (log)', linewidth=1.2, alpha=0.9)
            plt.xlabel('Epochs'); plt.ylabel('log(MSE)'); plt.title(f'Optimisation Curve (from epoch {plot_start}, log)')
            plt.legend(); plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'Optimisation_Curve_log.png')); plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(xs2, ys2, label='Training Loss', linewidth=1.2, alpha=0.9)
            plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.title(f'Optimisation Curve (from epoch {plot_start})')
            plt.legend(); plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'Optimisation_Curve.png')); plt.close()

            if sparse_ckpt_rows:
                pd.DataFrame(sparse_ckpt_rows).to_csv(
                    os.path.join(out_dir, "Holdout_MSE_sparse_at_checkpoints.csv"), index=False
                )

    return loss_history

'''
6. Prediction
'''
# def predict(model, data_list):
#     model.eval()
#     d = data_list[0].to(device)
#     with torch.no_grad():
#         return model(data_list).squeeze().cpu().numpy()
def predict(model, data_list, denorm=True):
    model.eval()
    with torch.no_grad():
        out = model(data_list).squeeze()
        if denorm:
            # factor = total_utility / N  (inverse of the training scale)
            d0 = data_list[0]
            N = int(d0.N.item()) if hasattr(d0, "N") else d0.x.size(0)
            tu = float(d0.total_util.item()) if torch.is_tensor(d0.total_util) else float(d0.total_util)
            factor = tu / N
            out = out * factor
        return out.cpu().numpy()


######################################################################################################
######################################################################################################
#########################################  Experiments ###############################################

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

import os, re

def discover_datasets(base_dir: str, require_tmc: bool = False):
    """
    Finds (train_like, test[, tmc]) triples under many naming schemes.

    Supported patterns (extension optional, any .csv/.parquet etc.):
      OLD:
        <stem>_<id>
        <stem>_test_<id>
        <stem>_TMC_<id>

      NEW (both positions supported):
        <stem>_<id>_train   or   <stem>_train_<id>
        <stem>_<id>_test    or   <stem>_test_<id>
        <stem>_<id>_TMC     or   <stem>_TMC_<id>
    """

    # compile once; allow any extension or none
    def rx(p):  # helper to make "(?:\.[^.]+)?$" optional extension
        return re.compile(p + r"(?:\.[^.]+)?$")

    R = {
        # TEST files
        "test_1":  rx(r"^(?P<stem>.+?)_test_(?P<id>\d+)"),
        "test_2":  rx(r"^(?P<stem>.+?)_(?P<id>\d+)_test"),
        # TRAIN-LIKE files
        "train_1": rx(r"^(?P<stem>.+?)_train_(?P<id>\d+)"),
        "train_2": rx(r"^(?P<stem>.+?)_(?P<id>\d+)_train"),
        # FALLBACK "train_like" (old bare style)
        "train_legacy": rx(r"^(?P<stem>.+?)_(?P<id>\d+)"),
        # TMC files
        "tmc_1":   rx(r"^(?P<stem>.+?)_TMC_(?P<id>\d+)"),
        "tmc_2":   rx(r"^(?P<stem>.+?)_(?P<id>\d+)_TMC"),
    }

    bucket = {}  # key: (stem, id) -> {"train_like":..., "test":..., "tmc":...}

    for fn in os.listdir(base_dir):
        path = os.path.join(base_dir, fn)
        base = os.path.splitext(fn)[0]

        matched = False
        # TEST
        for k in ("test_1", "test_2"):
            m = R[k].match(base)
            if m:
                key = (m.group("stem"), m.group("id"))
                bucket.setdefault(key, {})["test"] = path
                matched = True
                break
        if matched:
            continue

        # TRAIN-like (explicit _train_)
        for k in ("train_1", "train_2"):
            m = R[k].match(base)
            if m:
                key = (m.group("stem"), m.group("id"))
                bucket.setdefault(key, {})["train_like"] = path
                matched = True
                break
        if matched:
            continue

        # TMC
        for k in ("tmc_1", "tmc_2"):
            m = R[k].match(base)
            if m:
                key = (m.group("stem"), m.group("id"))
                bucket.setdefault(key, {})["tmc"] = path
                matched = True
                break
        if matched:
            continue

        # Legacy train_like fallback (<stem>_<id>) — but avoid swallowing _test_/_TMC_ forms we already handled
        m = R["train_legacy"].match(base)
        if m:
            key = (m.group("stem"), m.group("id"))
            # only accept as train_like if a more specific role wasn’t set by test/tmc above
            # and the filename didn’t actually contain "_test_" or "_TMC_" or "_train_"
            if "_test_" not in base and "_TMC_" not in base and "_train_" not in base:
                bucket.setdefault(key, {})["train_like"] = path
            continue

    # Build results
    results = []
    for (stem, idx), parts in bucket.items():
        tr = parts.get("train_like")
        te = parts.get("test")
        tm = parts.get("tmc")
        if tr and te and (tm or not require_tmc):
            base_name = f"{stem}_{idx}"  # canonical handle you can use in logs/plots
            results.append({
                "key": (stem, idx),
                "base_name": base_name,
                "train_like": tr,
                "test": te,
                "tmc": tm
            })
        else:
            # Helpful warning if something is missing
            missing = []
            if not tr: missing.append("train_like")
            if not te: missing.append("test")
            if require_tmc and not tm: missing.append("tmc")
            if missing:
                print(f"WARNING: incomplete set for {stem}_{idx}: missing {', '.join(missing)}")

    return results

def find_train_test_pairs(directory):
    test_re  = re.compile(r"^(.+?)_test_(\d+)$")
    train_re = re.compile(r"^(.+?)_(\d+)$")
    trains, tests = {}, {}
    for fn in os.listdir(directory):
        base = fn
        m = test_re.match(base)
        if m:
            key = (m.group(1), m.group(2))
            tests[key] = os.path.join(directory, fn)
            continue
        m = train_re.match(base)
        if m:
            key = (m.group(1), m.group(2))
            trains[key] = os.path.join(directory, fn)

    pairs = []
    for key, tr in trains.items():
        te = tests.get(key)
        if te:
            pairs.append((tr, te))
        else:
            print(f"⚠️  no test file for {tr!r}  (wanted {key[0]}_test_{key[1]})")
    return pairs

def _mk_paths(base_test_dir: str, base_name: str):
    parts = base_name.split("_")
    if len(parts) == 3:
        # MNIST-style: MNIST_{ab}_{i}
        prefix, ab, i = parts[0], parts[1], parts[2]
        train_like_csv = os.path.join(base_test_dir, base_name)
        test_csv       = os.path.join(base_test_dir, f"{prefix}_{ab}_test_{i}")
        tmc_csv        = os.path.join(base_test_dir, f"{prefix}_{ab}_TMC_{i}")
        return dict(test=test_csv, tmc=tmc_csv, train_like=train_like_csv)

    elif len(parts) == 2:
        # Two-part style: <prefix>_{i} (e.g., bc_1)
        prefix, i = parts[0], parts[1]
        train_like_csv = os.path.join(base_test_dir, base_name)
        test_csv       = os.path.join(base_test_dir, f"{prefix}_test_{i}")
        tmc_csv        = os.path.join(base_test_dir, f"{prefix}_TMC_{i}")
        return dict(test=test_csv, tmc=tmc_csv, train_like=train_like_csv)

    else:
        raise ValueError(f"Unexpected base_name format: {base_name}")

def print_adj_weights(model):
    with torch.no_grad():
        print("\n=== Learned adjacency weights per c in C_List ===")
        print(" idx | c  | alpha  beta  w_feature  w_label")
        for i, c in enumerate(C_List):
            a = model.alphas[i]
            b = model.betas[i]
            probs = F.softmax(torch.stack([a, b]), dim = 0)
            print(f"{i:>4d} | {c:<6} | {a.item():<13.6f} {b.item():<13.6f} {probs[0].item():<16.6f} {probs[1].item():<16.6f}")
    print()

COLORS = {
    "true": "tab:blue",      # True/Permuted Data Shapley
    "pred": "tab:purple",    # Predicted Data Shapley
    "tmc":  "tab:orange"     # TMC Data Shapley
}
from matplotlib.lines import Line2D
CLASS_LEGEND_HANDLES = [
    Line2D([0], [0], marker='o', linestyle='None', markersize=8,
           markerfacecolor='none', markeredgecolor='k', label='Class 0'),
    Line2D([0], [0], marker='x', linestyle='None', markersize=8,
           color='k', label='Class 1')
]

def diagnose_train_sets(gnn, base_train_dir: str, out_dir: str, graph_dir: str = None):
    _ensure_dir(out_dir)
    graph_dir = graph_dir or out_dir
    _ensure_dir(graph_dir)

    for it in discover_datasets(base_train_dir, require_tmc=False):
        train_csv, test_csv = it["train_like"], it["test"]
        dlist = load_and_preprocess(train_csv, test_csv)
        preds = predict(gnn, dlist)

    # pairs = find_train_test_pairs(base_train_dir)
    # for train_csv, test_csv in pairs:
    #     dlist = load_and_preprocess(train_csv, test_csv)
    #     preds = predict(gnn, dlist)

        df_train = pd.read_csv(train_csv)
        df_train['Pred_Shap'] = preds

        base_name = os.path.basename(train_csv).replace('.csv', '')
        out_csv   = os.path.join(out_dir, f'{base_name}_Pred.csv')
        df_train.to_csv(out_csv, index=False)
        print('saved ➜', out_csv)

        true_vals = df_train['PerDShap'].to_numpy()
        pred_vals = df_train['Pred_Shap'].to_numpy()
        y_labels = df_train['target'].to_numpy() if 'target' in df_train.columns else np.zeros_like(true_vals)
        mean_true, mean_pred = true_vals.mean(), pred_vals.mean()
        N = len(true_vals); idx = np.arange(N); width = 0.25

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # bar
        ax1.bar(idx - width, true_vals, width, label='True Shapley', color=COLORS["true"],
                edgecolor='k', alpha=0.7)
        ax1.bar(idx + width, pred_vals, width, label='Predicted Shapley', color=COLORS["pred"],
                edgecolor='k', alpha=0.7)
        ax1.plot(idx, [mean_true]*N, '--', linewidth=2, color=COLORS["true"],
                 label=f'Mean True: {mean_true:.3f}')
        ax1.plot(idx, [mean_pred]*N, '--', linewidth=2, color=COLORS["pred"],
                 label=f'Mean Pred: {mean_pred:.3f}')
        ax1.set_xlabel('Sample Index', fontsize=12, weight='bold')
        ax1.set_ylabel('Shapley Value', fontsize=12, weight='bold')
        ax1.set_title('Normalized True vs Predicted Shapley (Bar)', fontsize=13, weight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.4); ax1.legend(fontsize=10)

        # linear
        lo, hi = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())

        pad = 0.02 * (hi - lo if hi > lo else 1.0)
        ax2.set_xlim(lo - pad, hi + pad)
        ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_aspect('equal', adjustable='box')

        mask0 = (y_labels == 0)
        mask1 = (y_labels == 1)

        # class 0: hollow circle, class 1: cross
        ax2.scatter(true_vals[mask0], pred_vals[mask0],
                    s=80, facecolors='none', edgecolors=COLORS["pred"], linewidth=1.2,
                    marker='o', label='Pred vs True (class 0)')
        ax2.scatter(true_vals[mask1], pred_vals[mask1],
                    s=80, color=COLORS["pred"], marker='x', linewidth=1.2,
                    label='Pred vs True (class 1)')
        ax2.plot([lo, hi], [lo, hi], '--k', linewidth=1.5, label='y = x')

        ax2.set_xlabel('True Shapley Values (normalized)', fontsize=12, weight='bold')
        ax2.set_ylabel('Predicted Shapley Values', fontsize=12, weight='bold')
        ax2.set_title('True vs Predicted (Scatter)', fontsize=13, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4)
        legend1 = ax2.legend(fontsize=10, loc='upper left')
        ax2.add_artist(legend1)
        ax2.legend(handles=CLASS_LEGEND_HANDLES, fontsize=10, loc='lower right')

        fig.suptitle(f'Shapley Comparison: {os.path.basename(train_csv)}', fontsize=15, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(graph_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(out_png, dpi=300); plt.close()
        print('Saved to:', out_png)

def diagnose_test_sets(gnn, base_test_dir: str, holdout_bases, out_dir: str,
                       a=0.05, b=0.40, ylim=(40,105), random_iters=10):
    _ensure_dir(out_dir)
    catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc=False)}


    for base_name in holdout_bases:
        if base_name not in catalog:
            print(f"WARNING: '{base_name}' not found in {base_test_dir}.")
            continue

        paths = catalog[base_name]
        dlist = load_and_preprocess(paths["train_like"], paths["test"])
        preds = predict(gnn, dlist)
    #
    # for base_name in holdout_bases:
    #     paths = _mk_paths(base_test_dir, base_name)
    #
    #     dlist = load_and_preprocess(paths['train_like'], paths['test'])
    #     preds = predict(gnn, dlist)

        df_tr_like = pd.read_csv(paths['train_like'])
        df_tr_like['Pred_Shap'] = preds
        out_pred_csv = os.path.join(out_dir, f'{base_name}_Pred.csv')
        df_tr_like.to_csv(out_pred_csv, index=False)
        print('Saved to:', out_pred_csv)

        df_test = pd.read_csv(paths['test'])

        X_train = df_tr_like.drop(columns=['target','PerDShap','Pred_Shap']).astype(float)
        X_test  = df_test.drop(columns=['target']).astype(float)
        #X_test  = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        y_train = df_tr_like['target'].to_numpy()
        y_test  = df_test['target'].to_numpy()

        df_tmc   = pd.read_csv(paths['tmc'])
        TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
        Per_DShap = df_tr_like['PerDShap'].to_numpy()
        Pred_Shap = preds

        # Removal Plots
        desc_png = os.path.join(out_dir, f'{base_name}_Removal_Desc.png')
        _ = PerformancePlot_Removal(
            vals={'Permuted Data Shapley': Per_DShap,
                  'TMC Data Shapley': TMC_Shap,
                  'Predicted Shapley': Pred_Shap},
            X_train=X_train.to_numpy(), y_train=y_train,
            X_test=X_test.to_numpy(),   y_test=y_test,
            a=a, b=b, ylim=ylim,
            order='descending', random_iters=random_iters,
            save_path=desc_png
        )
        print('saved ➜', desc_png)

        asc_png = os.path.join(out_dir, f'{base_name}_Removal_Asc.png')
        _ = PerformancePlot_Removal(
            vals={'Permuted Data Shapley': Per_DShap,
                  'TMC Data Shapley': TMC_Shap,
                  'Predicted Shapley': Pred_Shap},
            X_train=X_train.to_numpy(), y_train=y_train,
            X_test=X_test.to_numpy(),   y_test=y_test,
            a=a, b=b, ylim=ylim,
            order='ascending', random_iters=random_iters,
            save_path=asc_png
        )
        print('Saved to:', asc_png)

        # Magnitude Comparison and Linear Comparison
        tmc_vals  = np.asarray(TMC_Shap)
        per_vals  = np.asarray(Per_DShap)
        pred_vals = np.asarray(Pred_Shap)
        mean_tmc, mean_per, mean_pred = tmc_vals.mean(), per_vals.mean(), pred_vals.mean()
        N = len(per_vals); idx = np.arange(N); width = 0.27

        plt.figure(figsize=(11,6))
        plt.bar(idx - width, per_vals,  width, label='Permuted Shapley', color=COLORS["true"],
                edgecolor='k', alpha=0.7)
        plt.bar(idx,          tmc_vals, width, label='TMC Shapley',      color=COLORS["tmc"],
                edgecolor='k', alpha=0.7)
        plt.bar(idx + width,  pred_vals, width, label='Predicted Shapley', color=COLORS["pred"],
                edgecolor='k', alpha=0.7)
        plt.plot(idx, [mean_per]*N,  '--', linewidth=2, color=COLORS["true"],
                 label=f'Avg Permuted ({mean_per:.3f})')
        plt.plot(idx, [mean_tmc]*N,  '--', linewidth=2, color=COLORS["tmc"],
                 label=f'Avg TMC ({mean_tmc:.3f})')
        plt.plot(idx, [mean_pred]*N, '--', linewidth=2, color=COLORS["pred"],
                 label=f'Avg Pred ({mean_pred:.3f})')
        plt.xlabel('Sample Index'); plt.ylabel('Shapley Value')
        plt.title(f'Comparison of Shapley Values — {base_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
        bar_png = os.path.join(out_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(bar_png, dpi=500); plt.close()
        print('Saved to:', bar_png)

        lo = min(per_vals.min(), tmc_vals.min(), pred_vals.min())
        hi = max(per_vals.max(), tmc_vals.max(), pred_vals.max())

        mask0 = (y_train == 0)
        mask1 = (y_train == 1)

        plt.figure(figsize=(8.8,8.8))

        # Pred vs Permuted (purple)
        plt.scatter(per_vals[mask0], pred_vals[mask0],
                    s=120, facecolors='none', edgecolors=COLORS["pred"], linewidth=1.3,
                    marker='o', label='Pred vs Permuted (class 0)')
        plt.scatter(per_vals[mask1], pred_vals[mask1],
                    s=120, color=COLORS["pred"], marker='x', linewidth=1.3,
                    label='Pred vs Permuted (class 1)')

        # TMC vs Permuted (orange)
        plt.scatter(per_vals[mask0], tmc_vals[mask0],
                    s=120, facecolors='none', edgecolors=COLORS["tmc"], linewidth=1.3,
                    marker='o', label='TMC vs Permuted (class 0)')
        plt.scatter(per_vals[mask1], tmc_vals[mask1],
                    s=120, color=COLORS["tmc"], marker='x', linewidth=1.3,
                    label='TMC vs Permuted (class 1)')

        plt.plot([lo - 0.005, hi + 0.005], [lo - 0.005, hi + 0.005],
                 '--k', linewidth=1.5, label='y = x')
        plt.xlabel('Permuted Shapley Value'); plt.ylabel('Other Shapley Values')
        plt.title(f'Linear Comparison — {base_name}')
        plt.grid(True, linestyle='--', alpha=0.5)

        leg1 = plt.legend(fontsize=10, loc='upper left')
        plt.gca().add_artist(leg1)
        plt.legend(handles=CLASS_LEGEND_HANDLES, fontsize=10, loc='lower right')

        plt.tight_layout()
        line_png = os.path.join(out_dir, f'{base_name}_LinearCompare.png')
        plt.savefig(line_png, dpi=500); plt.close()
        print('Saved to:', line_png)

def train_then_diagnose(BASE_TRAIN: str, BASE_TEST: str, OUT_DIR: str, holdout_bases,
                        in_channels=100, hidden_channels=128, heads=4,
                        epochs=1000, lr=5e-6, ckpt_every=10,
                        a=0.05, b=0.40, ylim=(40,105), random_iters=10):
    _ensure_dir(OUT_DIR)
    # pairs = find_train_test_pairs(BASE_TRAIN)
    # dataset_graphs = [load_and_preprocess(tr, te) for tr, te in pairs]
    train_triplets = discover_datasets(BASE_TRAIN, require_tmc=False)
    dataset_graphs = [load_and_preprocess(it["train_like"], it["test"]) for it in train_triplets]

    gnn = GINShap(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads)
    _ = train(gnn, dataset_graphs, epochs=epochs, lr=lr,
              base_test_dir=BASE_TEST, holdout_bases=holdout_bases, out_dir=OUT_DIR,
              ckpt_every=ckpt_every, checkpoint_dir=os.path.join(OUT_DIR, "checkpoints"),
              use_onecycle=True, max_lr=3e-4, onecycle_kwargs = dict(pct_start = 0.3, anneal_strategy = 'cos',
              cycle_momentum = True, base_momentum = 0.85, max_momentum = 0.95, div_factor = 25.0,
              final_div_factor = 1e4, three_phase = False))

    train_dir = os.path.join(OUT_DIR, "diagnostics_train_final"); _ensure_dir(train_dir)
    diagnose_train_sets(gnn, base_train_dir=BASE_TRAIN, out_dir=train_dir, graph_dir=train_dir)

    test_dir  = os.path.join(OUT_DIR, "diagnostics_test_final");  _ensure_dir(test_dir)
    diagnose_test_sets(gnn, base_test_dir=BASE_TEST, holdout_bases=holdout_bases,
                       out_dir=test_dir, a=a, b=b, ylim=ylim, random_iters=random_iters)
    return gnn



' To examine model performance at checkpoints'
def load_model_from_ckpt(ckpt_path, in_channels=100, hidden_channels=128, heads=4, strict=True):
    model = GINShap(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=strict)
    model.to(device).eval()
    print(f'loaded epoch {ckpt.get("epoch", "?" )} from {ckpt_path}')
    return model, ckpt.get('epoch', None)

def diagnose_from_checkpoints(ckpt_dir, epochs,
                              base_train_dir, base_test_dir, holdout_bases, out_root,
                              in_channels=100, hidden_channels=128, heads=4,
                              a=0.05, b=0.40, ylim=(40,105), random_iters=10):
    for ep in epochs:
        ck = os.path.join(ckpt_dir, f'model_ep{ep}.pt')
        if not os.path.exists(ck):
            print(f'Warning: checkpoint not found: {ck}')
            continue
        gnn_ep, loaded_ep = load_model_from_ckpt(ck, in_channels, hidden_channels, heads)

        out_dir_ep_train = os.path.join(out_root, f"diagnostics_train_ep{ep}")
        out_dir_ep_test  = os.path.join(out_root, f"diagnostics_test_ep{ep}")
        os.makedirs(out_dir_ep_train, exist_ok=True)
        os.makedirs(out_dir_ep_test, exist_ok=True)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TRAIN ===')
        diagnose_train_sets(gnn_ep, base_train_dir=base_train_dir,
                            out_dir=out_dir_ep_train, graph_dir=out_dir_ep_train)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TEST ===')
        diagnose_test_sets(gnn_ep, base_test_dir=base_test_dir, holdout_bases=holdout_bases,
                           out_dir=out_dir_ep_test, a=a, b=b, ylim=ylim, random_iters=random_iters)

        try:
            train_csv = os.path.join(out_root, "Training_MSE_full.csv")
            test_csv = os.path.join(out_root, "Holdout_MSE_full.csv")
            if os.path.exists(train_csv) and os.path.exists(test_csv):
                df_tr = pd.read_csv(train_csv)
                df_te = pd.read_csv(test_csv)
                # keep only selected epochs
                sel = pd.DataFrame({"epoch": epochs})
                df_tr_sel = sel.merge(df_tr, on="epoch", how="left")
                df_te_sel = sel.merge(df_te[["epoch", "avg_mse"]], on="epoch", how="left")

                plt.figure(figsize=(6, 4))
                plt.plot(df_te_sel["epoch"], df_te_sel["avg_mse"], label="Test (avg MSE)",
                         linewidth=1.2, alpha=0.9)  # lighter
                plt.plot(df_tr_sel["epoch"], df_tr_sel["train_mse"], label="Train (MSE)",
                         linewidth=1.2, alpha=0.9)  # lighter
                plt.xlabel("Epoch");
                plt.ylabel("MSE")
                plt.title("Train vs Test MSE — selected checkpoints")
                plt.grid(True, ls='--', alpha=0.5);
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_root, "Train_vs_Test_MSE_selected_epochs.png"))
                plt.close()
        except Exception as e:
            print("[warn] could not create Train_vs_Test_MSE_selected_epochs:", e)

def load_model_from_ckpt(ckpt_path, in_channels=100, hidden_channels=128, heads=4, strict=True):
    model = GINShap(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=strict)
    model.to(device).eval()
    ep = ckpt.get("epoch", None)
    print(f'loaded epoch {ep if ep is not None else "?"} from {ckpt_path}')
    return model, ep

def diagnose_from_checkpoints(ckpt_dir, epochs,
                              base_train_dir, base_test_dir, holdout_bases, out_root,
                              in_channels=100, hidden_channels=128, heads=4,
                              a=0.05, b=0.40, ylim=(40,105), random_iters=10):

    for ep in epochs:
        ck = os.path.join(ckpt_dir, f'model_ep{ep}.pt')
        if not os.path.exists(ck):
            print(f'⚠️ checkpoint not found: {ck}')
            continue
        gnn_ep, loaded_ep = load_model_from_ckpt(ck, in_channels, hidden_channels, heads)  # [FIX] unpack works

        out_dir_ep_train = os.path.join(out_root, f"diagnostics_train_ep{ep}")
        out_dir_ep_test  = os.path.join(out_root, f"diagnostics_test_ep{ep}")
        os.makedirs(out_dir_ep_train, exist_ok=True)
        os.makedirs(out_dir_ep_test, exist_ok=True)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TRAIN ===')
        diagnose_train_sets(gnn_ep, base_train_dir=base_train_dir,
                            out_dir=out_dir_ep_train, graph_dir=out_dir_ep_train)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TEST ===')
        diagnose_test_sets(gnn_ep, base_test_dir=base_test_dir, holdout_bases=holdout_bases,
                           out_dir=out_dir_ep_test, a=a, b=b, ylim=ylim, random_iters=random_iters)

    try:
        train_csv = os.path.join(out_root, "Training_MSE_full.csv")
        test_csv = os.path.join(out_root, "Holdout_MSE_full.csv")
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            # Load full histories once
            df_tr = pd.read_csv(train_csv).sort_values("epoch")  # columns: epoch, train_mse
            df_te = pd.read_csv(test_csv).sort_values("epoch")  # columns: epoch, ... , avg_mse

            # [ADD] For each requested checkpoint ep, plot from epoch 0 up to ep
            for ep in sorted(set(epochs)):
                df_tr_u = df_tr[df_tr["epoch"] <= ep].copy()
                df_te_u = df_te[df_te["epoch"] <= ep][["epoch", "avg_mse"]].copy()
                df_u = df_tr_u.merge(df_te_u, on="epoch", how="left")
                df_u = df_u[df_u["epoch"] >= 20]

                if df_u.empty:
                    print(f"[warn] no data to plot up to epoch {ep}")
                    continue

                plt.figure(figsize=(6, 4))
                plt.plot(df_u["epoch"], df_u["avg_mse"], label="Test (avg MSE)",
                         linewidth=1.2, alpha=0.9)  # lighter & visible
                plt.plot(df_u["epoch"], df_u["train_mse"], label="Train (MSE)",
                         linewidth=1.2, alpha=0.9)  # lighter & visible
                plt.xlabel("Epoch");
                plt.ylabel("MSE")
                plt.title(f"Train vs Test MSE — 0 to {ep} epochs")
                plt.grid(True, ls='--', alpha=0.5);
                plt.legend()
                plt.tight_layout()
                # [ADD] distinct file per checkpoint
                plt.savefig(os.path.join(out_root, f"Train_vs_Test_MSE_upto_epoch_{ep}.png"))
                plt.close()

            sel = pd.DataFrame({"epoch": epochs})
            df_tr_sel = sel.merge(df_tr, on="epoch", how="left")
            df_te_sel = sel.merge(df_te[["epoch", "avg_mse"]], on="epoch", how="left")

            plt.figure(figsize=(6, 4))
            plt.plot(df_te_sel["epoch"], df_te_sel["avg_mse"], label="Test (avg MSE)",
                     linewidth=1.2, alpha=0.9)
            plt.plot(df_tr_sel["epoch"], df_tr_sel["train_mse"], label="Train (MSE)",
                     linewidth=1.2, alpha=0.9)
            plt.xlabel("Epoch");
            plt.ylabel("MSE")
            plt.title("Train vs Test MSE — selected checkpoints")
            plt.grid(True, ls='--', alpha=0.5);
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_root, "Train_vs_Test_MSE_selected_epochs.png"))
            plt.close()
    except Exception as e:
        print("[warn] could not create cumulative Train/Test MSE plots:", e)


if __name__ == "__main__":
    BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/MNIST_Binary/Train'
    BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/MNIST_Binary/Test'
    OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251027/Trial02'
    HOLDOUTS   = ['MNIST_24xxx_14']

    # Train and Diagnosis
    _ = train_then_diagnose(
        BASE_TRAIN, BASE_TEST, OUT_DIR, HOLDOUTS,
        in_channels=100, hidden_channels=256, heads=8,
        epochs=200, ckpt_every=None
    )

    # # Manually output diagnosis from checkpoints
    # CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    # my_epochs = [70]
    # diagnose_from_checkpoints(
    #     ckpt_dir=CKPT_DIR, epochs=my_epochs,
    #     base_train_dir=BASE_TRAIN, base_test_dir=BASE_TEST,
    #     holdout_bases=HOLDOUTS, out_root=OUT_DIR,
    #     in_channels=10, hidden_channels=128, heads=4
    # )