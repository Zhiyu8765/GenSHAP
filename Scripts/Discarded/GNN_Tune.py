"""
LR Range Finder for OneCycle Policy
- Uses EXACTLY the same preprocessing, model, and target scaling as the main training script.
- Loss matches train(): MSE on scaled Shapley targets (phi * N / total_util).
- Plots match your style (dashed grid, linewidth=1.2, legend, plus a log-y companion plot).
"""

# =========================
# 0. Imports & device
# =========================
import os, re, math, random
from copy import deepcopy

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
from torch.nn import (Sequential, Linear, ReLU, TransformerEncoder,
                      TransformerEncoderLayer, LayerNorm)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Consistent color map with your other script
COLORS = {"true": "tab:blue", "pred": "tab:purple", "tmc": "tab:orange"}

# =========================
# 1. SVD (identical logic)
# =========================
def flip_pe_sign(x, method='sum'):
    assert method in ['none', 'pos', 'sum', 'l1', 'l2', 'linf', 'random']
    if method == 'none':
        return x
    elif method == 'random':
        sign = torch.randint(0, 2, (x.shape[1],), device=x.device) * 2 - 1
        return x * sign
    fliped_x = -1 * x
    if method == 'pos':
        x_v = torch.sum(x >= 0, dim=0); fliped_x_v = torch.sum(fliped_x >= 0, dim=0)
    elif method == 'sum':
        x_v = torch.sum(x, dim=0);       fliped_x_v = torch.sum(fliped_x, dim=0)
    elif method == 'linf':
        x_v, _ = torch.max(x, dim=0);    fliped_x_v, _ = torch.max(fliped_x, dim=0)
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
    sign = torch.sign(x_v - fliped_x_v)
    return x * sign

def SVD_Transform(A, d=100):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)  # A: (N, N)
    N = U.shape[0]
    k = min(N, d)
    X_svd = U[:, :k] * np.sqrt(S[:k])                 # (N, k)
    if k < d:
        pad = np.zeros((N, d - k), dtype=X_svd.dtype)
        X_svd = np.concatenate([X_svd, pad], axis=1)  # (N, d)
    X_t = torch.tensor(X_svd, dtype=torch.float, device=device)
    X_t = flip_pe_sign(X_t, method='sum')
    return X_t.cpu().numpy()                          # (N, d)

# =========================
# 2. Graph Construction
# =========================
C_List = [0.25, 0.5, 1, 2, 4]  # same as main script

def build_graph(X_raw, y_raw, phi, total_util, c, d=100):
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    X_svd = SVD_Transform(A, d=d)  # guaranteed (N, d)

    y = np.asarray(y_raw).ravel()
    A_y = (y[:, None] == y[None, :]).astype(float)

    N = X_raw.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(
        np.concatenate([eidx, eidx[::-1]], axis=1),
        dtype=torch.long, device=device
    )

    w_feat = A[src, dst]
    w_label = A_y[src, dst]
    edge_attr_feat = torch.tensor(
        np.concatenate([w_feat, w_feat]), dtype=torch.float, device=device
    ).unsqueeze(1)
    edge_attr_label = torch.tensor(
        np.concatenate([w_label, w_label]), dtype=torch.float, device=device
    ).unsqueeze(1)

    data = Data(
        x=torch.tensor(X_svd, dtype=torch.float, device=device),
        edge_index=edge_index,
        edge_attr_feat=edge_attr_feat,
        edge_attr_label=edge_attr_label
    )
    # Extras (same names/semantics as main script)
    data.X_raw = torch.tensor(X_raw, dtype=torch.float, device=device)
    data.phi = torch.tensor(phi, dtype=torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype=torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype=torch.long, device=device)
    data.num_classes = int(len(np.unique(y)))
    return data

# =========================
# 3. Data Preprocessing
# =========================
def load_and_preprocess(train_like_csv, test_csv, d=100):
    """
    Mirrors the main training script:
      - train_like_csv: features + 'target' + 'PerDShap'
      - test_csv      : features + 'target'
      - total_util    : LogisticRegression accuracy on test
      - returns: list[Data] (one per c in C_List)
    """
    df_train = pd.read_csv(train_like_csv)
    X_raw = df_train.drop(columns=['target', 'PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy()

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy().astype(int)

    lr_full = LogisticRegression(max_iter=100, random_state=23)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    pack = [build_graph(X_raw, y_raw, phi, total_util, d=d, c=c) for c in C_List]
    return pack

# --- discover datasets identical to main script ---
def discover_datasets(base_dir: str, require_tmc: bool = False):
    TEST_RE = re.compile(r"^(.+?)_test_(\d+)(?:\.[^.]+)?$")
    TMC_RE = re.compile(r"^(.+?)_TMC_(\d+)(?:\.[^.]+)?$")
    TRAIN_RE = re.compile(r"^(.+?)_(\d+)(?:\.[^.]+)?$")

    bucket = {}
    try:
        files = os.listdir(base_dir)
    except FileNotFoundError:
        files = []

    for fn in files:
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
    return result

# =========================
# 4. Model (Version A, identical)
# =========================
class GINShap(torch.nn.Module):
    """
    Version A: 5 separate GIN stacks, LayerNorm, TransformerEncoder, Z, and MLP(out).
    No extra injected vectors before MLP (matches the current training code).
    """
    def __init__(self, in_channels, hidden_channels=256, out_channels=1, heads=8):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))

        # Parallel branches
        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas  = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(3):
                convs.append(
                    GINConv(
                        mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                        aggregator_type='sum', init_eps=0.1, learn_eps=True, activation=ReLU()
                    )
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))

        d_concat = hidden_channels * len(C_List)
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first=True)
        self.gt = TransformerEncoder(encoder_layer, num_layers=3)

        self.Z = Sequential(Linear(d_concat, hidden_channels), ReLU())

        self.out = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, 100),
            ReLU(),
            Linear(100, out_channels)
        )

    def forward(self, data_list):
        data = data_list[0].to(device)
        x0 = data.x
        ei = data.edge_index

        # base dgl graph (recreated per pack anyway)
        g = dgl.graph((ei[0], ei[1]), num_nodes=x0.size(0))

        outputs = []
        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x.to(device)
            edge_attr_feat  = gdata.edge_attr_feat.view(-1).to(device)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label

            g = dgl.graph((gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
                          num_nodes=node_input.size(0), device=device)
            g.edata['w'] = edge_weight
            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])
            outputs.append(node_input)

        output_GIN  = torch.cat(outputs, dim=1)
        output_skip = output_GIN + self.skip_lin(x0)

        output_skip = output_skip.unsqueeze(0)
        output_gt   = self.gt(output_skip).squeeze(0)

        output_h = self.Z(output_gt)
        return self.out(output_h)  # (N, 1)

# =========================
# 5. LR Range Test (MSE on scaled φ)
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = float(lr)

@torch.no_grad()
def _mse_on_pack_scaled(model, dlist):
    """
    Matches train(): scale φ by N/total_util, compare to model output with MSE (mean).
    """
    model.eval()
    yhat_scaled = model(dlist).view(-1)
    total_util  = dlist[0].total_util.to(yhat_scaled.device)
    phi_true    = dlist[0].phi.to(yhat_scaled.device).view(-1)
    N           = phi_true.numel()
    scale       = N / total_util
    phi_true_sc = phi_true * scale
    mse = F.mse_loss(yhat_scaled, phi_true_sc, reduction="mean")
    model.train()
    return float(mse.item())

def smooth_curve(values, beta=0.98):
    smoothed = []
    avg = 0.0
    for i, v in enumerate(values, 1):
        avg = beta * avg + (1 - beta) * float(v)
        corrected = avg / (1 - beta**i)
        smoothed.append(corrected)
    return smoothed

def lr_range_test_linear_trainval_raw(
    model,
    train_packs,
    val_packs,
    lr_min=1e-7,
    lr_max=3e-3,
    num_iters=300,
    optimizer_ctor=torch.optim.AdamW,
    wd=0.0,
    out_dir='.',
    plot_name='LRFinder_TrainVal_MSE.png',
    plot_name_log='LRFinder_TrainVal_MSE_logy.png',
    csv_name='LRFinder_TrainVal_MSE.csv',
    verbose=True,
    smooth_beta=0.98,
):
    model = model.to(device).train()
    model_state = deepcopy(model.state_dict())
    optimizer = optimizer_ctor(model.parameters(), lr=lr_min, weight_decay=wd)

    if num_iters < 2:
        raise ValueError("num_iters must be >= 2")
    lrs = [lr_min + (lr_max - lr_min) * i / (num_iters - 1) for i in range(num_iters)]

    tr_idx = list(range(len(train_packs)))
    va_idx = list(range(len(val_packs)))
    random.shuffle(tr_idx); random.shuffle(va_idx)

    train_losses, val_losses = [], []

    for it, lr in enumerate(lrs, start=1):
        if it % len(tr_idx) == 0: random.shuffle(tr_idx)
        if it % len(va_idx) == 0: random.shuffle(va_idx)

        tr_pack = train_packs[tr_idx[(it - 1) % len(tr_idx)]]
        va_pack = val_packs[va_idx[(it - 1) % len(va_idx)]]

        _set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        # --- MSE on scaled φ (consistent with train()) ---
        yhat_tr      = model(tr_pack).view(-1)
        total_util_t = tr_pack[0].total_util.to(yhat_tr.device)
        phi_tr       = tr_pack[0].phi.to(yhat_tr.device).view(-1)
        N_tr         = phi_tr.numel()
        scale_tr     = N_tr / total_util_t
        phi_tr_sc    = phi_tr * scale_tr
        loss_tr      = F.mse_loss(yhat_tr, phi_tr_sc, reduction="mean")

        loss_tr.backward()
        optimizer.step()

        train_losses.append(float(loss_tr.item()))
        val_losses.append(_mse_on_pack_scaled(model, va_pack))

        if verbose and (it % max(1, num_iters // 10) == 0):
            print(f'[LR finder] step {it:4d}/{num_iters} | LR {lr:.2e} '
                  f'| train MSE {train_losses[-1]:.4g} | val MSE {val_losses[-1]:.4g}')

    tr_sm = smooth_curve(train_losses, beta=smooth_beta)
    va_sm = smooth_curve(val_losses,   beta=smooth_beta)

    _ensure_dir(out_dir)
    lrs_arr = np.array(lrs, dtype=float)

    # --- Plot (linear y): match your style ---
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(lrs_arr, train_losses, alpha=0.30, label='Train MSE (raw)', color=COLORS["true"])
    plt.plot(lrs_arr, val_losses,   alpha=0.30, label='Val MSE (raw)',   color=COLORS["pred"])
    plt.plot(lrs_arr, tr_sm,  linewidth=1.2, alpha=0.9, label='Train MSE (smoothed)', color=COLORS["true"])
    plt.plot(lrs_arr, va_sm,  linewidth=1.2, alpha=0.9, label='Val MSE (smoothed)',   color=COLORS["pred"])
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('MSE')
    plt.title('LR Range Test — Train vs Validation (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    fig_path = os.path.join(out_dir, plot_name)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print('LR finder plot saved ➜', fig_path)

    # --- Plot (log-y), like your Optimisation_Curve_log.png ---
    eps = 1e-12
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(lrs_arr, np.log(np.array(train_losses)+eps), linewidth=1.2, alpha=0.9,
             label='Train log(MSE)', color=COLORS["true"])
    plt.plot(lrs_arr, np.log(np.array(val_losses)+eps), linewidth=1.2, alpha=0.9,
             label='Val log(MSE)',   color=COLORS["pred"])
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('log(MSE)')
    plt.title('LR Range Test — log(MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    fig_path_log = os.path.join(out_dir, plot_name_log)
    plt.tight_layout()
    plt.savefig(fig_path_log, dpi=300)
    plt.close()
    print('LR finder log plot saved ➜', fig_path_log)

    df = pd.DataFrame({
        "lr": lrs_arr.tolist(),
        "train_mse": train_losses,
        "val_mse":   val_losses,
        "train_mse_smooth": tr_sm,
        "val_mse_smooth":   va_sm,
    })
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print('LR finder CSV saved  ➜', csv_path)

    # restore model weights
    model.load_state_dict(model_state)

    return {
        "lrs": lrs_arr.tolist(),
        "train_mse": train_losses,
        "val_mse":   val_losses,
        "train_mse_smooth": tr_sm,
        "val_mse_smooth":   va_sm,
        "plot_path": fig_path,
        "plot_path_log": fig_path_log,
        "csv_path":  csv_path,
    }

# =========================
# 6. Runner
# =========================
def run_lr_range_test(
    BASE_TRAIN: str,
    BASE_TEST: str,
    OUT_DIR: str,
    d=100,                 # fixed SVD width; must equal in_channels for the model
    hidden_channels=256,   # SAME as main script default
    heads=8,               # SAME as main script default
    c_list=(0.25, 0.5, 1, 2, 4),   # SAME as main script
    lr_min=1e-7,
    lr_max=3e-3,
    lr_num_iters=300,
    smooth_beta=0.98,
):
    global C_List
    C_List = list(c_list)
    _ensure_dir(OUT_DIR)

    # Build triplets exactly like the main script
    train_triplets = discover_datasets(BASE_TRAIN, require_tmc=False)
    val_triplets   = discover_datasets(BASE_TEST,  require_tmc=False)

    if not train_triplets:
        raise RuntimeError(f"No suitable (train_like, test) pairs found in {BASE_TRAIN}")
    if not val_triplets:
        raise RuntimeError(f"No suitable (train_like, test) pairs found in {BASE_TEST}")

    print(f'Found {len(train_triplets)} train pairs inside {BASE_TRAIN}.')
    print(f'Found {len(val_triplets)}  val pairs inside {BASE_TEST}.')

    # Build packs (each pack is a list[Data] with len(C_List))
    train_packs = [load_and_preprocess(it["train_like"], it["test"], d=d) for it in train_triplets]
    val_packs   = [load_and_preprocess(it["train_like"], it["test"], d=d) for it in val_triplets]

    # Model must receive in_channels == d
    gnn = GINShap(in_channels=d, hidden_channels=hidden_channels, heads=heads).to(device)

    # sanity for Transformer shape
    assert (hidden_channels * len(C_List)) % heads == 0, \
        "hidden_channels * len(C_List) must be divisible by heads"

    print('>>> Running LR range test (log-scale LR axis, MSE losses)...')
    res = lr_range_test_linear_trainval_raw(
        model=gnn,
        train_packs=train_packs,
        val_packs=val_packs,
        lr_min=lr_min,
        lr_max=lr_max,
        num_iters=lr_num_iters,
        optimizer_ctor=torch.optim.AdamW,
        wd=0.0,
        out_dir=OUT_DIR,
        plot_name='LRFinder_TrainVal_MSE.png',
        plot_name_log='LRFinder_TrainVal_MSE_logy.png',
        csv_name='LRFinder_TrainVal_MSE.csv',
        verbose=True,
        smooth_beta=smooth_beta,
    )
    print(f'Plot: {res["plot_path"]}')
    print(f'Plot (log): {res["plot_path_log"]}')
    print(f'CSV : {res["csv_path"]}')
    return res

# =========================
# 7. Example main
# =========================
if __name__ == "__main__":
    PARAMS = dict(
        BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/MNIST_Toy/Train',
        BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/1000pts_I/MNIST_Toy/Test',
        OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251026/Tuning_MNIST',

        d               = 100,     # fixed SVD width (== model in_channels)
        hidden_channels = 256,     # match main script
        heads           = 8,       # match main script
        c_list          = (0.25, 0.5, 1, 2, 4, 8),  # match main script

        lr_min       = 1e-6,
        lr_max       = 3e-3,
        lr_num_iters = 400,
        smooth_beta  = 0.98,
    )
    run_lr_range_test(**PARAMS)