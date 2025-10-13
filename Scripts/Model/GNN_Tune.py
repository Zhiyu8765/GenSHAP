'''
0. Import Packages
'''
import os, re, math, random
from copy import deepcopy

import dgl
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

'''
1. SVD
'''
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

def SVD_Transform(A, d=10):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    X_svd = U[:, :d] * np.sqrt(S[:d])
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = flip_pe_sign(X_t.to(device), method='sum')
    return X_t.cpu().numpy()

'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, y_raw, phi, total_util, c, d=10):
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    X_svd = SVD_Transform(A, d=d)

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
    data.X_raw = torch.tensor(X_raw, dtype=torch.float, device=device)
    data.phi = torch.tensor(phi, dtype=torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype=torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype=torch.long, device=device)
    return data

'''
3. Data Preprocessing
'''
def load_and_preprocess(train_like_csv, test_csv, d=10):
    """
    train_like_csv must contain: features, 'target', 'PerDShap'
    test_csv must contain: features, 'target'
    """
    df_train = pd.read_csv(train_like_csv)
    if 'PerDShap' not in df_train.columns:
        raise RuntimeError(f"'PerDShap' missing in {train_like_csv}")
    X_raw = df_train.drop(columns=['target', 'PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy()

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy()

    lr_full = LogisticRegression(max_iter=100, random_state=23)
    lr_full.fit(X_raw, y_raw)
    total_util = lr_full.score(X_test, y_test)

    pack = [build_graph(X_raw, y_raw, phi, total_util, d=d, c=c) for c in C_List]
    return pack

'Datasets Matching'
TRAIN_RE = re.compile(r"^(.+?)_(\d+)(?:\.csv)?$", re.IGNORECASE)
TEST_RE  = re.compile(r"^(.+?)_test_(\d+)(?:\.csv)?$", re.IGNORECASE)
TMC_RE   = re.compile(r"^(.+?)_tmc_(\d+)(?:\.csv)?$", re.IGNORECASE)

def _list_files(dirpath):
    try:
        return [f for f in os.listdir(dirpath)]
    except FileNotFoundError:
        return []

def find_triplets_within(dirpath: str):

    files = _list_files(dirpath)
    trains, tests = {}, {}
    for fn in files:
        if TEST_RE.match(fn):
            m = TEST_RE.match(fn); tests[(m.group(1), m.group(2))] = os.path.join(dirpath, fn)
        elif TRAIN_RE.match(fn) and not TMC_RE.match(fn):
            m = TRAIN_RE.match(fn); trains[(m.group(1), m.group(2))] = os.path.join(dirpath, fn)

    pairs = []
    for key, tr in trains.items():
        te = tests.get(key)
        if te:
            pairs.append((tr, te))
        else:
            print(f"⚠️  Missing test for {tr!r} (expected {key[0]}_test_{key[1]} in {dirpath})")
    return pairs


'''
4. Model Construction
'''
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=1, heads=8):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LayerNorm(o), ReLU(), Linear(o, o))
        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas  = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(3):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels,
                                hidden_channels),
                            aggregator_type='sum', init_eps=0.1, learn_eps=True,
                            activation=ReLU())
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))
        d_concat = hidden_channels * len(C_List)
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first=True)
        self.gt = TransformerEncoder(encoder_layer, num_layers=3)
        self.Z  = Sequential(Linear(d_concat, hidden_channels), ReLU())
        self.out = Sequential(
            Linear(hidden_channels + 2, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, 100),
            ReLU(),
            Linear(100, out_channels)
        )

    def forward(self, data_list):
        data = data_list[0].to(device)
        x0 = data.x
        ei = data.edge_index

        g = dgl.graph((ei[0], ei[1]), num_nodes=x0.size(0), device=device)

        outputs = []
        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x.to(device)
            edge_attr_feat  = gdata.edge_attr_feat.view(-1).to(device)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label

            g = dgl.graph((gdata.edge_index[0], gdata.edge_index[1]),
                          num_nodes=node_input.size(0), device=device)
            g.edata['w'] = edge_weight
            for conv in convs:
                node_input = conv(g, node_input, edge_weight=g.edata['w'])
            outputs.append(node_input)

        h = torch.cat(outputs, dim=1)
        h = h + self.skip_lin(x0)
        h = h.unsqueeze(0)
        h = self.gt(h).squeeze(0)
        h = self.Z(h)

        N = h.size(0)
        total_util = data.total_util
        utility_vec = total_util.expand(N, 1)
        NU_vector = torch.full((N, 1), np.sqrt(N) / 10., device=device)
        h_final = torch.cat([h, utility_vec, NU_vector], dim=1)
        return self.out(h_final)

'''
5. LR Range Test
'''
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = float(lr)

@torch.no_grad()
def _loss_on_pack(model, dlist):
    model.eval()
    preds  = model(dlist).view(-1)
    target = dlist[0].phi.to(device)
    loss = F.mse_loss(preds, target, reduction='mean')
    model.train()
    return float(loss.item())

def smooth_curve(values, beta=0.98):
    # EMA
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
    plot_name='LRFinder_Linear_TrainVal.png',
    csv_name='LRFinder_Linear_TrainVal.csv',
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
        preds_tr  = model(tr_pack).view(-1)
        target_tr = tr_pack[0].phi.to(device)
        loss_tr   = F.mse_loss(preds_tr, target_tr, reduction='mean')
        loss_tr.backward()
        optimizer.step()

        train_losses.append(float(loss_tr.item()))
        val_losses.append(_loss_on_pack(model, va_pack))

        if verbose and (it % max(1, num_iters // 10) == 0):
            print(f'[LR finder] step {it:4d}/{num_iters} | LR {lr:.2e} '
                  f'| train {train_losses[-1]:.4g} | val {val_losses[-1]:.4g}')

    tr_sm = smooth_curve(train_losses, beta=smooth_beta)
    va_sm = smooth_curve(val_losses,   beta=smooth_beta)

    _ensure_dir(out_dir)
    lrs_arr = np.array(lrs, dtype=float)

    plt.figure(figsize=(8.0, 5.2))
    plt.plot(lrs_arr, train_losses, alpha=0.25, label='Train loss (raw)')
    plt.plot(lrs_arr, val_losses,   alpha=0.25, label='Val loss (raw)')
    plt.plot(lrs_arr, tr_sm,  linewidth=2.0, label='Train loss (smoothed)')
    plt.plot(lrs_arr, va_sm,  linewidth=2.0, label='Val loss (smoothed)')
    plt.yscale('log')
    plt.xlabel('Learning Rate (linear ramp)')
    plt.ylabel('Loss (MSE)')
    plt.title('LR Range Test — Train vs Validation Loss (raw + EMA)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    fig_path = os.path.join(out_dir, plot_name)
    plt.savefig(fig_path, dpi=240, bbox_inches='tight')
    plt.close()
    print('LR finder plot saved ➜', fig_path)

    df = pd.DataFrame({
        "lr": lrs_arr.tolist(),
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_loss_smooth": tr_sm,
        "val_loss_smooth":   va_sm,
    })
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print('LR finder CSV saved  ➜', csv_path)

    model.load_state_dict(model_state)

    return {
        "lrs": lrs_arr.tolist(),
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_loss_smooth": tr_sm,
        "val_loss_smooth":   va_sm,
        "plot_path": fig_path,
        "csv_path":  csv_path,
    }

def run_lr_range_test(
    BASE_TRAIN: str,
    BASE_TEST: str,
    OUT_DIR: str,
    in_channels=10,
    hidden_channels=128,
    heads=4,
    d=10,
    c_list=(0.25, 0.5, 1, 2, 4),
    lr_min=1e-7,
    lr_max=3e-3,
    lr_num_iters=300,
    smooth_beta=0.98,
):
    global C_List
    C_List = list(c_list)
    _ensure_dir(OUT_DIR)

    # Discover triplets
    train_pairs = find_triplets_within(BASE_TRAIN)
    if len(train_pairs) == 0:
        print('Files in BASE_TRAIN:')
        for f in _list_files(BASE_TRAIN): print('  ', f)
        raise RuntimeError(f"No triplets found in BASE_TRAIN: {BASE_TRAIN}")

    val_pairs = find_triplets_within(BASE_TEST)
    if len(val_pairs) == 0:
        print('Files in BASE_TEST:')
        for f in _list_files(BASE_TEST): print('  ', f)
        raise RuntimeError(f"No triplets found in BASE_TEST: {BASE_TEST}")

    print(f'Found {len(train_pairs)} train triplets inside {BASE_TRAIN}.')
    print(f'Found {len(val_pairs)}  val triplets inside {BASE_TEST}.')
    print(f'Using {len(train_pairs)} train packs and {len(val_pairs)} val packs (holdout only).')

    # Build packs
    train_packs = [load_and_preprocess(tr, te, d=d) for tr, te in train_pairs]
    val_packs   = [load_and_preprocess(tr, te, d=d) for tr, te in val_pairs]

    # Model
    gnn = GINShap(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads).to(device)

    # LR finder (raw + smoothed)
    print('>>> Running LR range test (linear ramp, TRAIN & VAL losses)...')
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
        plot_name='LRFinder_Linear_TrainVal.png',
        csv_name='LRFinder_Linear_TrainVal.csv',
        verbose=True,
        smooth_beta=smooth_beta,
    )
    print(f'Plot: {res["plot_path"]}')
    print(f'CSV : {res["csv_path"]}')
    return res

if __name__ == "__main__":
    PARAMS = dict(
        BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialE_MNIST/Train',
        BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialE_MNIST/Test',
        OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250928/LR_RT3',

        in_channels     = 10,
        hidden_channels = 128,
        heads           = 4,
        d               = 10,
        c_list          = (0.25, 0.5, 1, 2, 4),

        lr_min       = 1e-7,
        lr_max       = 1e-2,
        lr_num_iters = 300,

        smooth_beta  = 0.98,
    )
    run_lr_range_test(**PARAMS)