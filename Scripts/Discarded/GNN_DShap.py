'''
251023 Updated: Modifications included:
1) Now support multi-class classification;
2) Scaling on Data Shapley value in training process;
3) SSE are used instead of MSE;
4) Removed additionally injected vectors before MLP
'''


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

from matplotlib.lines import Line2D
from itertools import cycle
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
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


'''
1. SVD
'''
'Flip Sign'
def flip_pe_sign(x, method='l0'):
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

svd_d = 50
FEAT_DIM = svd_d
def SVD_Transform(A, d = svd_d):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(d, U.shape[1])
    X_svd = U[:, :k] * np.sqrt(S[:k])
    if k < d:
        pad = np.zeros((X_svd.shape[0], d - k), dtype = X_svd.dtype)
        X_svd = np.concatenate([X_svd, pad], axis = 1)
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = flip_pe_sign(X_t, method='sum')  # l0 (positive) / sum
    return X_t.numpy()



'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, y_raw, phi, total_util, c, d = svd_d):
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

    data = Data(
        x = torch.tensor(X_svd, dtype = torch.float, device=device),
        edge_index = edge_index,
        edge_attr_feat = edge_attr_feat,
        edge_attr_label = edge_attr_label
    )

    # Extra attributes
    data.X_raw = torch.tensor(X_raw, dtype = torch.float, device=device)
    data.phi = torch.tensor(phi, dtype = torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype = torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype = torch.long, device=device)
    data.num_classes = int(len(np.unique(y)))

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

    data_list = [] # length = 6
    for c in C_List:
        data = build_graph(X_raw, y_raw, phi, total_util, d = d, c = c)
        data_list.append(data)

    return data_list


'''
4. Model Construction
'''
'Version A: With LayerNorm & 5 separate conv stacks'
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels = 256, out_channels = 1, heads = 8):
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
            Linear(hidden_channels, hidden_channels * 2),
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

            g = dgl.graph(
                (gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
                num_nodes=node_input.size(0),
                device=device
            )
            g.edata['w'] = edge_weight

            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])
            output_GIN.append(node_input)

        # Skip-Connection
        output_GIN = torch.cat(output_GIN, dim = 1)
        output_skip = output_GIN + self.skip_lin(x0)

        # GT
        output_skip = output_skip.unsqueeze(0) # transform to: (1, N, hidden_channels*6)
        output_gt = self.gt(output_skip)
        output_gt = output_gt.squeeze(0)

        # H
        output_h = self.Z(output_gt)

        # total utility + (N/10) -> MLP
        # N = output_h.size(0)
        # total_util = data.total_util
        # utility_vec = total_util.expand(N, 1)
        # NU_vector = torch.full((N, 1), np.sqrt(N)/10., device = device)
        # output_final = torch.cat([output_h, utility_vec, NU_vector], dim = 1)

        return self.out(output_h)

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
'Training with OneCycle Policy'
def train(model, dataset_graphs, epochs=500, lr=1e-5, weight_decay = 1e-7,
          base_test_dir: str = None, holdout_bases: list = None, out_dir: str = None,
          ckpt_every: int = 10, checkpoint_dir: str = None, use_onecycle: bool = False,
          max_lr: float = None, onecycle_kwargs: dict = None):
    model.to(device)

    if out_dir:
        os.makedirs(out_dir, exist_ok = True)
    if checkpoint_dir is None and out_dir:
        checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok = True)

    init_lr = lr
    if use_onecycle and max_lr is not None and onecycle_kwargs and "div_factor" in onecycle_kwargs:
        init_lr = float(max_lr)/float(onecycle_kwargs["div_factor"])
    optimizer = torch.optim.AdamW(model.parameters(), lr = init_lr, weight_decay = weight_decay)

    scheduler = None
    if use_onecycle:
        if max_lr is None:
            raise ValueError("max_lr cannot be None")
        steps_per_epoch = max(1, len(dataset_graphs))
        total_steps = epochs * steps_per_epoch

        oc_defaults = dict(pct_start = 0.3, anneal_strategy = 'cos',
                           cycle_momentum = True, base_momentum = 0.85, max_momentum = 0.95,
                           div_factor = 25.0, final_div_factor = 1e4, three_phase = False)
        if onecycle_kwargs:
            oc_defaults.update(onecycle_kwargs)

        scheduler = OneCycleLR(optimizer, max_lr = max_lr, total_steps = total_steps, **oc_defaults)

    loss_history = []
    per_epoch_rows = []
    sparse_ckpt_rows = []

    def _current_lr(opt):
        return float(opt.param_groups[0]['lr'])

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for data_list in dataset_graphs:
            total_util = data_list[0].total_util.to(device)
            phi_true = data_list[0].phi.to(device).view(-1)
            N = phi_true.numel()
            scale = N/total_util
            phi_true_scaled = phi_true * scale
            phi_hat_scaled = model(data_list).view(-1)
            loss = torch.sum((phi_hat_scaled - phi_true_scaled) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            losses.append(loss.detach())

        L = torch.stack(losses).mean()
        loss_history.append(L.item())

        if ep % 5 == 0:
            print(f'Epoch {ep:3d} | total_loss{L:.6e} | lr: {_current_lr(optimizer):.6e}')

        avg_sse = None
        if holdout_bases and base_test_dir:
            model.eval()
            row = {"epoch": ep}
            with torch.no_grad():
                catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc = False)}
                sses = []
                for base in holdout_bases:
                    if base not in catalog:
                        print(f"WARNING: Not found in {base_test_dir}: {base}")
                        continue
                    p = catalog[base]
                    dl = load_and_preprocess(p['train_like'], p['test'], d = model.skip_lin.in_features)
                    pred_scaled = model(dl).view(-1)
                    phi_t = dl[0].phi.to(device).view(-1)
                    total_util_t = dl[0].total_util.to(device)
                    N_t = phi_t.numel()
                    scale_t = (N_t/total_util_t)
                    phi_t_scaled = phi_t * scale_t
                    m = torch.sum((pred_scaled - phi_t_scaled) ** 2).item()
                    row[base] = m
                    sses.append(m)
                if sses:
                    avg_sse = float(np.mean(sses))
                    row['avg_sse'] = avg_sse
                    per_epoch_rows.append(row)
                    print(f'>>> Average SSE in test set at {ep}: {avg_sse:.10f}')

        if ckpt_every and (ep % ckpt_every == 0) and checkpoint_dir:
            path = os.path.join(checkpoint_dir, f"model_ep{ep}.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss_train_sse": L.item(),
                "avg_test_sse": avg_sse,
            }, path)
            print("Checkpoint saved to:", path)
            if avg_sse is not None:
                sparse_ckpt_rows.append({"epoch": ep, "avg_sse": avg_sse})

    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, f"model_ep{ep}.pt")
        torch.save({"epochs": epochs, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "loss_train_sse": L.item(), "avg_test_sse": (per_epoch_rows[-1]["avg_sse"] if per_epoch_rows else None)}
                   , final_path)
        print("Final model saved to:", final_path)

    if out_dir:
        xs = np.arange(len(loss_history))
        ys = np.array(loss_history)
        pd.DataFrame({"epoch": xs + 1, "train_sse": ys}).to_csv(os.path.join(out_dir, "Training_SSE.csv"), index = False)

        if per_epoch_rows:
            df_full = pd.DataFrame(per_epoch_rows)
            csv_full = os.path.join(out_dir, "Holdout_SSE_Full.csv")
            df_full.to_csv(csv_full, index = False); print("saved to:", csv_full)

            df_train_full = pd.DataFrame({"epoch": xs + 1, "train_sse": ys})
            df_tt = df_full.merge(df_train_full, on = "epoch", how = "left")
            df_tt = df_tt[df_tt["epoch"] >= 20]

            plt.figure(figsize = (6, 4))
            plt.plot(df_tt["epoch"], df_tt["avg_sse"], label = "Test(avg SSE)", linewidth = 1.2, alpha = 0.9)
            plt.plot(df_tt["epoch"], df_tt["train_sse"], label = "Train (avg SSE)", linewidth = 1.2, alpha = 0.9)
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
            plt.xlabel('Epochs');
            plt.ylabel('log(MSE)');
            plt.title(f'Optimisation Curve(log)')
            plt.legend();
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout();
            plt.savefig(os.path.join(out_dir, 'Optimisation_Curve_log.png'));
            plt.close()

            if sparse_ckpt_rows:
                pd.DataFrame(sparse_ckpt_rows).to_csv(os.path.join(out_dir, "Sparse_CKPT.csv"), index = False)

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
        tr = paths.get("train_like");
        te = paths.get("test");
        tm = paths.get("tmc")
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
            a = model.alphas[i]
            b = model.betas[i]
            probs = F.softmax(torch.stack([a, b]), dim = 0)
            print(f"{i:>4d} | {c:<6} | {a.item():<13.6f}  {b.item():<13.6f}  {probs[0].item():<16.6f}  {probs[1].item():<16.6f}")
        print()

COLORS = {"true": "tab:blue", "pred": "tab:purple", "tmc": "tab:orange"}

from itertools import cycle
def _class_styles(unique_labels):
    FILLED = ['o','s','^','D','P','*','v','>','<','H']
    it = cycle(FILLED)
    styles = {}
    for cls in sorted(unique_labels):
        mk = next(it)
        # Do NOT include facecolors here
        styles[cls] = dict(marker=mk, linewidths=1.2)
    return styles

def diagnose_train_sets(gnn, base_train_dir: str, out_dir: str, graph_dir: str = None):
    _ensure_dir(out_dir)
    graph_dir = graph_dir or out_dir
    _ensure_dir(graph_dir)

    triplets = discover_datasets(base_train_dir, require_tmc = False)
    for item in triplets:
        train_csv, test_csv = item["train_like"], item["test"]
        dlist = load_and_preprocess(train_csv, test_csv, d = gnn.skip_lin.in_features)
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
        N = len(true_vals)
        idx = np.arange(N)
        width = 0.25

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

        # --- Bar plot ---
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

        # --- Scatter: class -> marker, all points purple ---
        lo, hi = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
        pad = 0.02 * (hi - lo if hi > lo else 1.0)
        ax2.set_xlim(lo - pad, hi + pad)
        ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_aspect('equal', adjustable='box')

        classes = np.unique(y_labels)
        styles = _class_styles(classes)

        # Scatter all in purple; marker shape encodes class
        for c in classes:
            mask = (y_labels == c)
            st = styles[c].copy()
            ax2.scatter(
                true_vals[mask], pred_vals[mask],
                **st,
                facecolors='none',
                edgecolors=COLORS["pred"],
                linewidths=1.2
            )

        # Diagonal
        ax2.plot([lo, hi], [lo, hi], '--k', linewidth=1.5)

        ax2.set_xlabel('True Shapley Values (normalized)', fontsize=12, weight='bold')
        ax2.set_ylabel('Predicted Shapley Values', fontsize=12, weight='bold')
        ax2.set_title('True vs Predicted (Scatter)', fontsize=13, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4)

        # Class legend (marker only)
        class_handles = []
        for c in classes:
            st = styles[c]
            if st.get('facecolors') == 'none':
                class_handles.append(
                    Line2D([0],[0], marker=st['marker'], linestyle='None',
                           markerfacecolor='none', markeredgecolor='k',
                           markersize=8, label=f'class {c}')
                )
            else:
                class_handles.append(
                    Line2D([0],[0], marker=st['marker'], linestyle='None',
                           markerfacecolor='k', markeredgecolor='k',
                           markersize=8, label=f'class {c}')
                )
        ax2.legend(handles=class_handles, title="Class (marker)", fontsize=9, loc='best')

        fig.suptitle(f'Shapley Comparison: {os.path.basename(train_csv)}', fontsize=15, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(graph_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(out_png, dpi=300); plt.close()
        print('Saved to:', out_png)

def diagnose_test_sets(gnn, base_test_dir: str, holdout_bases, out_dir: str, a = 0.05, b = 0.40,
                      ylim = (40, 105), random_iters = 10):
    _ensure_dir(out_dir)

    catalog = {it["base_name"]: it for it in discover_datasets(base_test_dir, require_tmc = False)}
    for base in holdout_bases:
        if base not in catalog:
            print(f"WARNING: Base {base} not found in {base_test_dir}.")
            continue
        paths = catalog[base]

        dlist = load_and_preprocess(paths["train_like"], paths["test"], d = gnn.skip_lin.in_features)
        preds = predict(gnn, dlist)

        df_tr_like = pd.read_csv(paths['train_like'])
        df_tr_like['Pred_Shap'] = preds
        out_pred_csv = os.path.join(out_dir, f'{base}_Pred.csv')
        df_tr_like.to_csv(out_pred_csv, index = False)
        print('Saved to:', out_pred_csv)

        df_test = pd.read_csv(paths['test'])
        X_train = df_tr_like.drop(columns=['target','PerDShap','Pred_Shap']).astype(float)
        X_test  = df_test.drop(columns=['target']).astype(float)
        y_train = df_tr_like['target'].to_numpy().astype(int)
        y_test  = df_test['target'].to_numpy().astype(int)

        if paths.get('tmc') is None:
            print(f"WARNING: No TMC file for {base}.")
            continue

        df_tmc   = pd.read_csv(paths['tmc'])
        TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
        Per_DShap = df_tr_like['PerDShap'].to_numpy()
        Pred_Shap = preds

        # --- Removal Plots (unchanged) ---
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

        # --- Bar comparison (unchanged) ---
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
        plt.title(f'Comparison of Shapley Values — {base}')
        plt.grid(axis='y', linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
        bar_png = os.path.join(out_dir, f'{base}_ShapleyCompare.png')
        plt.savefig(bar_png, dpi=500); plt.close()
        print('saved ➜', bar_png)

        # --- Linear comparison: series color + class marker ---
        lo = min(per_vals.min(), tmc_vals.min(), pred_vals.min())
        hi = max(per_vals.max(), tmc_vals.max(), pred_vals.max())

        classes = np.unique(y_train)
        styles = _class_styles(classes)

        fig, ax = plt.subplots(figsize=(8.8, 8.8))

        # Predicted (purple) and TMC (orange) with the SAME marker per class
        for c in classes:
            mask = (y_train == c)
            st = styles[c].copy()
            ax.scatter(per_vals[mask], pred_vals[mask],
                       **st, facecolors='none', edgecolors=COLORS["pred"], linewidths=1.2)
            ax.scatter(per_vals[mask], tmc_vals[mask],
                       **st, facecolors='none', edgecolors=COLORS["tmc"], linewidths=1.2)

        # Diagonal y = x
        ax.plot([lo - 0.005, hi + 0.005], [lo - 0.005, hi + 0.005], '--k', linewidth=1.5)

        ax.set_xlabel('Permuted Shapley Value')
        ax.set_ylabel('Other Shapley Values')
        ax.set_title(f'Linear Comparison — {base}')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Legends
        # 1) Series (color) legend
        series_handles = [
            Line2D([0],[0], marker='o', linestyle='None',
                   markerfacecolor='none', markeredgecolor=COLORS["pred"], markersize=8, label='Predicted Shapley'),
            Line2D([0],[0], marker='o', linestyle='None',
                   markerfacecolor='none', markeredgecolor=COLORS["tmc"], markersize=8, label='TMC Shapley'),
        ]
        legend_series = ax.legend(handles=series_handles, title="Series (color)", loc='upper left', fontsize=9)
        ax.add_artist(legend_series)

        # 2) Class (marker) legend
        class_handles = []
        for c in classes:
            st = styles[c]
            if st.get('facecolors') == 'none':
                class_handles.append(
                    Line2D([0],[0], marker=st['marker'], linestyle='None',
                           markerfacecolor='none', markeredgecolor='k',
                           markersize=8, label=f'class {c}')
                )
            else:
                class_handles.append(
                    Line2D([0],[0], marker=st['marker'], linestyle='None',
                           markerfacecolor='k', markeredgecolor='k',
                           markersize=8, label=f'class {c}')
                )
        legend_classes = ax.legend(handles=class_handles, title="Class (marker)", loc='lower right', fontsize=9)
        ax.add_artist(legend_classes)

        plt.tight_layout()
        line_png = os.path.join(out_dir, f'{base}_LinearCompare.png')
        plt.savefig(line_png, dpi=500); plt.close()
        print('Saved to:', line_png)

def train_then_diagnose(BASE_TRAIN: str, BASE_TEST: str, OUT_DIR: str, holdout_bases,
                        in_channels = svd_d, hidden_channels = 256, heads = 8,
                        epochs = 500, lr = 1e-5, weight_decay = 1e-7,
                        ckpt_every = 10, a = 0.05, b = 0.40, ylim = (40, 105), random_iters = 10):
    _ensure_dir(OUT_DIR)

    triplets = discover_datasets(BASE_TRAIN, require_tmc = False)
    dataset_graphs = [load_and_preprocess(it["train_like"], it["test"], d = in_channels) for it in triplets]

    gnn = GINShap(in_channels = in_channels, hidden_channels = hidden_channels, heads = heads)
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

def load_model_from_ckpt(ckpt_path, in_channels = svd_d, hidden_channels = 256, heads = 8, strict = True):
    model = GINShap(in_channels = in_channels, hidden_channels = hidden_channels, heads = heads)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model"))
    model.load_state_dict(state, strict = strict)
    model.to(device).eval()
    print(f'loaded epoch {ckpt.get("epoch", "?")} from {ckpt_path}')
    return model, ckpt.get('epoch', None)

def diagnose_from_checkpoints(ckpt_dir, epochs, base_train_dir, base_test_dir, holdout_bases, out_root,
                              in_channels = svd_d, hidden_channels = 256, heads = 8,
                              a = 0.05, b = 0.4, ylim = (40, 105), random_iters = 10):
    for ep in epochs:
        ck = os.path.join(ckpt_dir, f'model_ep{ep}.pt')
        if not os.path.exists(ck):
            print(f'WARNING: checkpoint not found: {ck}')
            continue
        gnn_ep, loaded_ep = load_model_from_ckpt(ck, in_channels, hidden_channels, heads)

        out_dir_ep_train = os.path.join(out_root, f"diagnostics_train_ep{ep}")
        out_dir_ep_test = os.path.join(out_root, f"diagnostics_test_ep{ep}")
        os.makedirs(out_dir_ep_train, exist_ok=True)
        os.makedirs(out_dir_ep_test, exist_ok=True)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TRAIN ===')
        diagnose_train_sets(gnn_ep, base_train_dir=base_train_dir,
                            out_dir=out_dir_ep_train, graph_dir=out_dir_ep_train)

        print(f'=== Diagnostics @ epoch {ep} (loaded {loaded_ep}) : TEST ===')
        diagnose_test_sets(gnn_ep, base_test_dir=base_test_dir, holdout_bases=holdout_bases,
                           out_dir=out_dir_ep_test, a=a, b=b, ylim=ylim, random_iters=random_iters)

        try:
            train_csv = os.path.join(out_root, "Training_SSE_full.csv")
            test_csv = os.path.join(out_root, "Holdout_SSE_full.csv")
            if os.path.exists(train_csv) and os.path.exists(test_csv):
                df_tr = pd.read_csv(train_csv)
                df_te = pd.read_csv(test_csv)
                sel = pd.DataFrame({"epoch": epochs})
                df_tr_sel = sel.merge(df_tr, on="epoch", how="left")
                df_te_sel = sel.merge(df_te[["epoch", "avg_sse"]], on="epoch", how="left")

                plt.figure(figsize=(6, 4))
                plt.plot(df_te_sel["epoch"], df_te_sel["avg_sse"], label="Test (avg MSE)",
                         linewidth=1.2, alpha=0.9)
                plt.plot(df_tr_sel["epoch"], df_tr_sel["train_sse"], label="Train (MSE)",
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
            print("WARNING: Could not create Train_vs_Test_MSE_selected_epochs:", e)






######################################################################################################
######################################################################################################
#########################################  Experiments ###############################################

if __name__ == "__main__":
    BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/Train'
    BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/Test'
    OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251025/Trial01'
    HOLDOUTS   = ['Iris_1']

    # Train and Diagnosis
    _ = train_then_diagnose(
        BASE_TRAIN, BASE_TEST, OUT_DIR, HOLDOUTS,
        in_channels=100, hidden_channels=256, heads=8,
        epochs=10, ckpt_every=None
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