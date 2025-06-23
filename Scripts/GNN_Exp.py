'''
0. Import Packages
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.extmath import randomized_svd
from tensorflow.python.ops.gen_nn_ops import LeakyRelu
from torch_geometric.nn import TransformerConv, GINConv, global_add_pool
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout

'''
1. Randomized SVD
'''
def filp_pe_sign(x, method='l0'):
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

def SVD_Transform(A, d=10):
    U, S, _ = randomized_svd(A, n_components=d, random_state=42) #change to full svd/torch svd
    X_svd = U * np.sqrt(S)
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = filp_pe_sign(X_t, method='l1') # l0 (positive) / sum
    return X_t.numpy()

'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, d = 10, tmc = None, c = 0.25):
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    A = np.maximum(A, A.T) #(delete)
    X_svd = SVD_Transform(A, d = d)

    N = A.shape[0]
    upper_tri = np.triu_indices(N, k = 1)
    src, dst = upper_tri
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = A[src, dst]
    edge_attr = torch.tensor(edge_weight, dtype = torch.float).unsqueeze(1)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    data = Data(
        x = torch.tensor(X_svd, dtype = torch.float),
        edge_index = edge_index, edge_attr = edge_attr
    )
    data.X_raw = torch.tensor(X_raw, dtype = torch.float)
    if tmc is not None:
        data.tmc = torch.tensor(tmc, dtype = torch.float)

    return data

'''
3. Data Preprocessing
'''
def load_and_preprocess(csv_path, d=10):
    df = pd.read_csv(csv_path)
    X_raw = pd.get_dummies(df.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
    phi = df['TrueDShap'].to_numpy()
    N = len(phi)
    phi_scaled = phi * N

    data_list = []
    for c in C_List:
        data = build_graph(X_raw = X_raw, d = d, tmc = phi_scaled, c = c)
        data_list.append(data)

    return data_list

'''
4. Model Construction
'''
### Without Parallel GIN
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128,
                 out_channels=1, heads=2, dropout=0.3):
        super().__init__()

        # 带权重的0阶矩阵 -》 DGL
        def mlp(i, o):
            return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))

        # 加上skip-connection
        self.conv1 = GINConv(mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(mlp(hidden_channels, hidden_channels))

        self.H = Sequential(
            Linear(hidden_channels, hidden_channels), LeakyReLU(),
            Linear(hidden_channels, hidden_channels), LeakyReLU(),
            Linear(hidden_channels, hidden_channels), LeakyReLU()
        )

        self.gt1 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1) #用torch的transformerencoder
        self.gt2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1)
        self.gt3 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1)

        self.out = Sequential(
            Linear(hidden_channels, hidden_channels), LeakyReLU(),
            Dropout(dropout),
            Linear(hidden_channels, hidden_channels), LeakyReLU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, d):
        x = self.conv1(d.x, d.edge_index) # 用上边的权重
        x = self.conv2(x, d.edge_index)
        x = self.conv3(x, d.edge_index)

        x = self.H(x)

        x = self.gt1(x, d.edge_index, d.edge_attr)
        x = self.gt2(x, d.edge_index, d.edge_attr)
        x = self.gt3(x, d.edge_index, d.edge_attr)

        return self.out(x)


# ### With Parallel GIN
# class MultiGraphGINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=256,
#                  out_channels=1, heads=2, dropout=0.5):
#         super().__init__()
#
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))
#
#         # 5 parallel branches of GIN
#         self.Parallel_GINs = torch.nn.ModuleList([
#             torch.nn.ModuleList([
#                 GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels))
#                 for layer in range(3)
#             ])
#             for _ in C_List
#         ])
#
#         # GIN
#         self.Merge_GIN = torch.nn.ModuleList([
#             GINConv(mlp(hidden_channels * len(C_List) if layer == 0 else hidden_channels, hidden_channels))
#             for layer in range(3)
#         ])
#
#         # H
#         self.H = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.LeakyReLU())
#
#         # GT
#         self.gt    = torch.nn.ModuleList([
#             TransformerConv(hidden_channels, hidden_channels//heads, heads = heads, edge_dim = 1)
#             for _ in range(3)
#         ])
#
#         # Z
#         self.Z = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.LeakyReLU())
#
#         # MLP
#         self.out   = torch.nn.Sequential(
#             torch.nn.Linear(hidden_channels, hidden_channels),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(hidden_channels, hidden_channels),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(hidden_channels, out_channels),
#         )
#
#     def forward(self, data_list):
#         xs = []
#         for branch, data in zip(self.Parallel_GINs, data_list):
#             x = data.x
#             for conv in branch:
#                 x = conv(x, data.edge_index)
#             xs.append(x)
#         x = torch.cat(xs, dim = 1)
#         for conv in self.Merge_GIN:
#             x = conv(x, data_list[0].edge_index)
#         x = self.H(x)
#         for gt in self.gt:
#             x = gt(x, data_list[0].edge_index, data_list[0].edge_attr)
#         x = self.Z(x)
#         return self.out(x)
#
#
# ### With Parallel GIN (Batch Normalisation Added)
# from torch.nn import LayerNorm
# class MultiGraphGINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=256,
#                  out_channels=1, heads=2, dropout=0.4):
#         super().__init__()
#
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))
#
#         self.Parallel_GINs = torch.nn.ModuleList([
#             torch.nn.ModuleList([
#                 GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels))
#                 for layer in range(3)
#             ]) for _ in C_List
#         ])
#
#         self.Parallel_BNs = torch.nn.ModuleList([
#             torch.nn.ModuleList([
#                 LayerNorm(hidden_channels) for _ in range(3)
#             ]) for _ in C_List
#         ])
#
#         self.Merge_GIN = torch.nn.ModuleList([
#             GINConv(mlp(hidden_channels * len(C_List) if layer == 0 else hidden_channels, hidden_channels))
#             for layer in range(3)
#         ])
#         self.Merge_BNs = torch.nn.ModuleList([
#             LayerNorm(hidden_channels) for _ in range(3)
#         ])
#
#         self.H = Sequential(Linear(hidden_channels, hidden_channels), LeakyReLU())
#
#         self.gt = torch.nn.ModuleList([
#             TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1)
#             for _ in range(3)
#         ])
#         self.gt_BNs = torch.nn.ModuleList([
#             LayerNorm(hidden_channels) for _ in range(3)
#         ])
#
#         self.Z = Sequential(Linear(hidden_channels, hidden_channels), LeakyReLU())
#
#         self.out = Sequential(
#             Linear(hidden_channels, hidden_channels),
#             LeakyReLU(),
#             Dropout(dropout),
#             Linear(hidden_channels, hidden_channels),
#             LeakyReLU(),
#             Dropout(dropout),
#             Linear(hidden_channels, out_channels),
#         )
#
#     def forward(self, data_list):
#         xs = []
#         for branch, bn_branch, data in zip(self.Parallel_GINs, self.Parallel_BNs, data_list):
#             x = data.x
#             for conv, bn in zip(branch, bn_branch):
#                 x = conv(x, data.edge_index)
#                 x = bn(x)
#                 x = F.leaky_relu(x)
#             xs.append(x)
#
#         x = torch.cat(xs, dim=1)
#
#         for conv, bn in zip(self.Merge_GIN, self.Merge_BNs):
#             x = conv(x, data_list[0].edge_index)
#             x = bn(x)
#             x = F.leaky_relu(x)
#
#         x = self.H(x)
#
#         for gt, bn in zip(self.gt, self.gt_BNs):
#             x = gt(x, data_list[0].edge_index, data_list[0].edge_attr)
#             x = bn(x)
#             x = F.leaky_relu(x)
#
#         x = self.Z(x)
#         return self.out(x)


'''
5. Training
'''
def train(model, dataset_graphs, epochs=200, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # adamw
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # cosine

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        per_node_losses = []

        for data_list in dataset_graphs:
            d = data_list[0].to(device)
            phi_hat = model(d).squeeze()
            phi_true = d.tmc.to(device)

            mae_loss = F.l1_loss(phi_hat, phi_true, reduction='mean')
            loss_i = mae_loss
            per_node_losses.append(loss_i)

        epoch_loss = torch.stack(per_node_losses).mean()
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        avg_loss_ep = epoch_loss.item()
        if ep % 5 == 0:
            print(f'Epoch {ep:3d} | total loss {avg_loss_ep:.4f}| LR: {scheduler.get_last_lr()[0]:.2e}')

'''
8. Prediction
'''
def predict(model, data_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    d = data_list[0].to(device)
    with torch.no_grad():
        return model(d).squeeze().cpu().numpy()


######################################################################################################
######################################################################################################
#########################################  Experiments ###############################################

'''
EXP 1. Iris (Subsets of label 1 & 2 for testing)
'''

'A. Implementation'
if __name__ == '__main__':
    #BASE = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets'
    BASE = '/tmp/pycharm_project_236/Datasets'
    #PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612'
    PATH_OUT = '/tmp/pycharm_project_236/Outputs/250619/Trial05'
    Train_list = [
        (f'{BASE}/Iris_01'),
        (f'{BASE}/Iris_02'),
        (f'{BASE}/bc_1'),
        (f'{BASE}/bc_2'),
        (f'{BASE}/bc_3'),
        (f'{BASE}/MNIST_01'),
        (f'{BASE}/MNIST_12'),
        (f'{BASE}/MNIST_23'),
        (f'{BASE}/MNIST_34'),
        (f'{BASE}/MNIST_45'),
        (f'{BASE}/MNIST_56'),
        (f'{BASE}/MNIST_67')
    ]
    holdout_train_csv = f'{BASE}/Iris_12'
    holdout_test_csv = f'{BASE}/TestIris_12'

    data_list = [
        load_and_preprocess(train_csv)
        for train_csv in Train_list
    ]
    data_holdout = load_and_preprocess(holdout_train_csv)

    # all_phi_scaled = torch.cat([d.tmc.view(-1) for d in data_list], dim = 0)
    # bias_init = float(all_phi_scaled.mean().item())

    gnn = GINShap(in_channels=10, hidden_channels=256)
    train(gnn, data_list, epochs=200, lr=1e-3)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['Pred_Shap'] = preds
    out_csv = f'{PATH_OUT}/Iris_12_Pred.csv'
    df_holdout.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

    # Predict and save for each training dataset
    for train_csv, data in zip(Train_list, data_list):
        preds_train = predict(gnn, data)

        df_train = pd.read_csv(train_csv)
        df_train['Pred_Shap'] = preds_train

        base_name = os.path.basename(train_csv)
        out_csv_train = os.path.join(PATH_OUT, f'{base_name}_Pred.csv')
        df_train.to_csv(out_csv_train, index=False)
        print('saved ➜', out_csv_train)


'Removal Plots'
#df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TestIris_12')
df_test = pd.read_csv('/tmp/pycharm_project_236/Datasets/TestIris_12')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
#df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Iris_12_TMC')
df_train = pd.read_csv('/tmp/pycharm_project_236/Datasets/Iris_12_TMC')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','TrueDShap','TMC_Shap'])).to_numpy()
y_train = df_train['target'].to_numpy()
True_Shap = df_train['TrueDShap'].to_numpy()
N = len(True_Shap)
True_Shap = True_Shap * N
TMC_Shap = df_train['TMC_Shap'].to_numpy()
TMC_Shap = TMC_Shap * N
Pred_Shap = preds
PERF_PATH = os.path.join('/tmp/pycharm_project_236/Outputs/250619/Trial05', 'Iris12Pred_High.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap,
          'Predicted Shapley': Pred_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.01, b = 0.5, ylim = (20,105),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)


'Distribution/Magnitude Comparison'
#GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612'
GRAPH_DIR = '/tmp/pycharm_project_236/Outputs/250619/Trial05'
true_vals = np.asarray(True_Shap)
tmc_vals  = np.asarray(TMC_Shap)
pred_vals = np.asarray(Pred_Shap)
mean_true = true_vals.mean()
mean_tmc  = tmc_vals.mean()
mean_pred = pred_vals.mean()
N = len(true_vals)
indices = np.arange(N)
width = 0.25
plt.figure(figsize=(10,6))
plt.bar(indices - width, true_vals,  width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
plt.bar(indices, tmc_vals, width, label='TMC Shapley', color='tab:orange',edgecolor='k', alpha=0.7)
plt.bar(indices + width, pred_vals, width, label='Predicted Shapley',  color='tab:green', edgecolor='k', alpha=0.7)
plt.plot(indices, [mean_true]*N, linestyle='--', linewidth=2, label=f'Average True Shapley Values ({mean_true:.3f})',   color='tab:blue')
plt.plot(indices, [mean_tmc]*N, linestyle='--', linewidth=2, label=f'Average TMC Shapley Values ({mean_tmc:.3f})',     color='tab:orange')
plt.plot(indices, [mean_pred]*N, linestyle='--', linewidth=2, label=f'Average Predicted Shapley Values ({mean_pred:.3f})',color='tab:green')
plt.xlabel('Sample Index', fontsize=12, weight='bold')
plt.ylabel('Shapley Value', fontsize=12, weight='bold')
plt.title('Comparison of True, TMC & Predicted Shapley Values', fontsize=14, weight='bold')
plt.xticks(indices)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, 'Shapley_compare.png')
plt.savefig(out_path, dpi=500)
plt.show()

'Linear Plot'
true_vals = np.asarray(True_Shap)
pred_vals = np.asarray(Pred_Shap)
tmc_vals  = np.asarray(TMC_Shap)

plt.figure(figsize=(8,8))
plt.scatter(true_vals, pred_vals,
            s=200, color='tab:blue', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='Predicted Shapley')
plt.scatter(true_vals, tmc_vals,
            s=200, color='tab:orange', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='TMC Shapley')
all_min = min(true_vals.min(), pred_vals.min(), tmc_vals.min())
all_max = max(true_vals.max(), pred_vals.max(), tmc_vals.max())
plt.plot([all_min, all_max], [all_min, all_max],
         '--k', linewidth=1.5, label='y = x')

plt.xlabel('True TMC Shapley Values', fontsize=10, weight='bold')
plt.ylabel('Predicted TMC Shapley Values', fontsize=10, weight='bold')
plt.title('True vs Predicted & TMC Shapley Values', fontsize=12, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

out_path = os.path.join(GRAPH_DIR, 'DShap_LinearCompare.png')
plt.tight_layout()
plt.show()
plt.savefig(out_path, dpi=500)


'Check Train Sets'
BASE2 = '/tmp/pycharm_project_236/Outputs/250619/Trial03'
Train_list = [
        (f'{BASE2}/Iris_01_Pred.csv'),
        (f'{BASE2}/Iris_02_Pred.csv'),
        (f'{BASE2}/bc_1_Pred.csv'),
        (f'{BASE2}/bc_2_Pred.csv'),
        (f'{BASE2}/bc_3_Pred.csv'),
        (f'{BASE2}/MNIST_01_Pred.csv'),
        (f'{BASE2}/MNIST_12_Pred.csv'),
        (f'{BASE2}/MNIST_23_Pred.csv'),
        (f'{BASE2}/MNIST_34_Pred.csv'),
        (f'{BASE2}/MNIST_45_Pred.csv'),
        (f'{BASE2}/MNIST_56_Pred.csv'),
        (f'{BASE2}/MNIST_67_Pred.csv')
    ]
for train in Train_list:
    df = pd.read_csv(train)
    true_vals = df['TrueDShap']
    scaler_true = MinMaxScaler()
    True_Shap = scaler_true.fit_transform(true_vals.to_numpy().reshape(-1, 1)).flatten()
    true_vals = np.asarray(True_Shap)

    pred_vals = df['Pred_Shap']
    pred_vals = np.asarray(pred_vals)

    mean_true = true_vals.mean()
    mean_pred = pred_vals.mean()
    N = len(true_vals)
    indices = np.arange(N)
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot
    ax1.bar(indices - width, true_vals, width, label='True Shapley', color='tab:blue', edgecolor='k', alpha=0.7)
    ax1.bar(indices + width, pred_vals, width, label='Predicted Shapley', color='tab:green', edgecolor='k', alpha=0.7)
    ax1.plot(indices, [mean_true] * N, '--', linewidth=2, color='tab:blue',
             label=f'Mean True: {mean_true:.3f}')
    ax1.plot(indices, [mean_pred] * N, '--', linewidth=2, color='tab:green',
             label=f'Mean Pred: {mean_pred:.3f}')
    ax1.set_xlabel('Sample Index', fontsize=12, weight='bold')
    ax1.set_ylabel('Shapley Value', fontsize=12, weight='bold')
    ax1.set_title('Normalized True vs Predicted Shapley (Bar)', fontsize=13, weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(fontsize=10)
    ax2.scatter(true_vals, pred_vals, s=80, color='tab:blue', alpha=0.6,
                edgecolors='k', linewidth=0.5, label='Predicted vs True')
    all_min = min(true_vals.min(), pred_vals.min())
    all_max = max(true_vals.max(), pred_vals.max())
    ax2.plot([all_min, all_max], [all_min, all_max], '--k', linewidth=1.5, label='y = x')
    ax2.set_xlabel('True Shapley Values (normalized)', fontsize=12, weight='bold')
    ax2.set_ylabel('Predicted Shapley Values', fontsize=12, weight='bold')
    ax2.set_title('True vs Predicted (Scatter)', fontsize=13, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=10)

    fig.suptitle(f'Shapley Comparison: {os.path.basename(train)}', fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

    out_name = os.path.basename(train).replace('.csv', '') + '_ShapleyCompare.png'
    out_path = os.path.join(GRAPH_DIR, out_name)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print('saved ➜', out_path)





'''
9. Implementation
'''
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'
    Train_list = [
        (f'{BASE}/train1.csv'),
        (f'{BASE}/train2.csv'),
        (f'{BASE}/train4.csv'),
        (f'{BASE}/train5.csv'),
        (f'{BASE}/train6.csv')
    ]
    holdout_train_csv = f'{BASE}/train3.csv'
    holdout_test_csv = f'{BASE}/test.csv'

    data_list = [
        load_and_preprocess(train_csv)
        for train_csv in Train_list
    ]
    data_holdout = load_and_preprocess(holdout_train_csv)

    all_phi_scaled = torch.cat([d.tmc.view(-1) for d in data_list], dim = 0)
    bias_init = float(all_phi_scaled.mean().item())

    gnn = GINShap(data_list[0].num_node_features, hidden_channels=1024)
    train(gnn, data_list, epochs=300, lr=1e-4)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/train3_predicted(5+1)_15.csv'
    df_holdout.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

'''
10. Performance Plots
'''
'Removal Plots'
df_test = pd.read_csv(holdout_test_csv)
X_test = pd.get_dummies(df_test.drop(columns = ['label'])).to_numpy()
y_test = df_test['label'].to_numpy()

X_train=pd.get_dummies(df_holdout.drop(
                columns=['label','G_Shap','LOO','TMC_Shap','predicted_TMC_Shap'])).to_numpy()
y_train = df_holdout['label'].to_numpy()
TMC_true = df_holdout['TMC_Shap'].to_numpy()
TMC_true = TMC_true * len(TMC_true)
LOO_vals = df_holdout['LOO'].to_numpy()
G_true = df_holdout['G_Shap'].to_numpy()
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605', 'AIDS3pred_(5+1)_low_15.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_true,
          'predicted TMC': preds,
          'G Shapley': G_true,
          'LOO': LOO_vals},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (20,85),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)

'Distribution/Magnitude Comparison'
DATA_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'

TRAIN3_CSV = os.path.join(DATA_DIR, 'train3.csv')
Train_3 = pd.read_csv(TRAIN3_CSV)
TMC_train3 = Train_3['TMC_Shap'].values
TMC_train3 = len(TMC_train3) * TMC_train3

TMC_Shap_AIDS_Pred = preds

avg_s = np.mean(TMC_Shap_AIDS_Pred)
avg_1 = np.mean(TMC_train3)

x_s = np.arange(len(TMC_Shap_AIDS_Pred))
x_1 = np.arange(len(TMC_train3))
plt.figure(figsize = (10, 6))

plt.stem(x_s, TMC_Shap_AIDS_Pred, linefmt = 'tab:blue', markerfmt = '', basefmt = '', label = 'Predicted Shapley Values')
plt.axhline(avg_s, color = 'tab:blue', linestyle = '--', linewidth = 2, label = 'Average Predicted Values')
plt.stem(x_1, TMC_true, linefmt = 'tab:orange', markerfmt = '', basefmt = '', label = 'TMC Shapley Values')
plt.axhline(avg_1, color = 'tab:orange', linestyle = '--', linewidth = 2, label = 'Average TMC Shapley Values')

plt.xlabel('Training Data Index', fontsize = 14, weight = 'bold')
plt.ylabel('Data Shapley Values', fontsize = 14, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'TMC_Comparison15.png'))

'Linear Plot'
true_vals = np.asarray(TMC_train3)
pred_vals = np.asarray(TMC_Shap_AIDS_Pred)
plt.figure(figsize = (8,8))
plt.scatter(true_vals, pred_vals, s = 30, color = 'tab:blue', alpha = 0.6, edgecolors = 'k',
            linewidth = 0.5, label = 'Predicted TMC Shapley Values vs True TMC Shapley Values')
all_min = min(true_vals.min(), pred_vals.min())
all_max = max(true_vals.max(), pred_vals.max())
plt.plot([all_min, all_max], [all_min, all_max], '--k', linewidth = 1.5, label = 'y=x')
plt.xlabel('True TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.ylabel('Predicted TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.title('True vs Predicted TMC Shapley Values', fontsize = 12, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.legend(fontsize = 10)
out_path = os.path.join(GRAPH_DIR, 'TMC_LinearCompare15.png')
plt.tight_layout()
plt.savefig(out_path, dpi = 500)
plt.close()
print(f'Saved')


'''
9. Implementation: MNIST
'''
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'
    Train_Test_Pairs = [
        (f'{BASE}/train01.csv', f'{BASE}/test01.csv'),
        (f'{BASE}/train12.csv', f'{BASE}/test12.csv'),
        (f'{BASE}/train23.csv', f'{BASE}/test23.csv'),
        (f'{BASE}/train34.csv', f'{BASE}/test34.csv')
    ]
    holdout_train_csv = f'{BASE}/train45.csv'
    holdout_test_csv = f'{BASE}/test45.csv'

    data_list = [
        load_and_preprocess(train_csv, test_csv)
        for train_csv, test_csv in Train_Test_Pairs
    ]
    data_holdout = load_and_preprocess(holdout_train_csv, holdout_test_csv)


    gnn = GINShap(data_list[0].num_node_features, hidden_channels=512)
    train(gnn, data_list, epochs=100, lr=1e-4)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/trainMNIST45_predicted1.csv'
    df_holdout.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

'''
10. Performance Plots
'''
df_test = pd.read_csv(holdout_test_csv)
X_test = pd.get_dummies(df_test.drop(columns = ['label'])).to_numpy()
y_test = df_test['label'].to_numpy()

X_train=pd.get_dummies(df_holdout.drop(
                columns=['label','G_Shap','LOO','TMC_Shap','predicted_TMC_Shap'])).to_numpy()
y_train = df_holdout['label'].to_numpy()
TMC_true = df_holdout['TMC_Shap'].to_numpy()
TMC_true = TMC_true * len(TMC_true)
LOO_vals = df_holdout['LOO'].to_numpy()
G_true = df_holdout['G_Shap'].to_numpy()

PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515', 'MNIST45pred_High_1.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_true,
          'predicted TMC': preds,
          'G Shapley': G_true,
          'LOO': LOO_vals},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,54),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)
'Distribution/Magnitude Comparison'
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'
TMC_pred = preds

avg_pred = np.mean(TMC_pred)
avg_true = np.mean(TMC_true)

x_pred = np.arange(len(TMC_pred))
x_true = np.arange(len(TMC_true))
plt.figure(figsize = (10, 6))

plt.stem(x_pred, TMC_pred, linefmt = 'tab:blue', markerfmt = '', basefmt = '', label = 'Predicted Shapley Values')
plt.axhline(avg_pred, color = 'tab:blue', linestyle = '--', linewidth = 2, label = 'Average Predicted Values')
plt.stem(x_true, TMC_true, linefmt = 'tab:orange', markerfmt = '', basefmt = '', label = 'True TMC Shapley Values')
plt.axhline(avg_true, color = 'tab:orange', linestyle = '--', linewidth = 2, label = 'Average TMC Shapley Values')

plt.xlabel('Training Data Index', fontsize = 14, weight = 'bold')
plt.ylabel('Data Shapley Values', fontsize = 14, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'TMC_ComparisonMNIST45_1.png'))

'Linear Plot'
true_vals = np.asarray(TMC_true)
pred_vals = np.asarray(TMC_pred)
plt.figure(figsize = (8,8))
plt.scatter(true_vals, pred_vals, s = 30, color = 'tab:blue', alpha = 0.6, edgecolors = 'k',
            linewidth = 0.5, label = 'Predicted TMC Shapley Values vs True TMC Shapley Values')
all_min = min(true_vals.min(), pred_vals.min())
all_max = max(true_vals.max(), pred_vals.max())
plt.plot([all_min, all_max], [all_min, all_max], '--k', linewidth = 1.5, label = 'y=x')
plt.xlabel('True TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.ylabel('Predicted TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.title('True vs Predicted TMC Shapley Values', fontsize = 12, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.legend(fontsize = 10)
out_path = os.path.join(GRAPH_DIR, 'TMC_LinearCompareMNIST45_1.png')
plt.tight_layout()
plt.savefig(out_path, dpi = 500)
plt.close()
print(f'Saved')


'''
9. Implementation: AIDS + MNIST
'''
if __name__ == '__main__':
    BASE1 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets'
    BASE2 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'
    Train_list = [
        (f'{BASE1}/train1.csv'),
        (f'{BASE1}/train2.csv'),
        (f'{BASE1}/train4.csv'),
        (f'{BASE1}/train5.csv'),
        (f'{BASE1}/train6.csv'),
        (f'{BASE2}/train01.csv'),
        (f'{BASE2}/train12.csv'),
        (f'{BASE2}/train23.csv'),
        (f'{BASE2}/train34.csv'),
        (f'{BASE2}/train45.csv'),
    ]
    holdout_train_csv = f'{BASE1}/train3.csv'
    holdout_test_csv = f'{BASE1}/test.csv'

    data_list = [
        load_and_preprocess(train_csv)
        for train_csv in Train_list
    ]
    data_holdout = load_and_preprocess(holdout_train_csv)

    gnn = GINShap(in_channels=30, hidden_channels=512)
    train(gnn, data_list, epochs=300, lr=1e-4)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/train3_pred(10+1)_2.csv'
    df_holdout.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

'''
10. Performance Plots
'''
df_test = pd.read_csv(holdout_test_csv)
X_test = pd.get_dummies(df_test.drop(columns = ['label'])).to_numpy()
y_test = df_test['label'].to_numpy()

X_train=pd.get_dummies(df_holdout.drop(
                columns=['label','G_Shap','LOO','TMC_Shap','predicted_TMC_Shap'])).to_numpy()
y_train = df_holdout['label'].to_numpy()
TMC_true = df_holdout['TMC_Shap'].to_numpy()
TMC_true = TMC_true * len(TMC_true)
LOO_vals = df_holdout['LOO'].to_numpy()
G_true = df_holdout['G_Shap'].to_numpy()

PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605', 'AIDS3pred(10+1)_high_1.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_true,
          'predicted TMC': preds,
          'G Shapley': G_true,
          'LOO': LOO_vals},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,85),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)

'Read in Data Shapley Values of train1'
DATA_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250605'

TRAIN3_CSV = os.path.join(DATA_DIR, 'train3.csv')
Train_3 = pd.read_csv(TRAIN3_CSV)
TMC_train3 = Train_3['TMC_Shap'].values
TMC_train3 = len(TMC_train3) * TMC_train3

TMC_Shap_AIDS_Pred = preds

avg_s = np.mean(TMC_Shap_AIDS_Pred)
avg_1 = np.mean(TMC_train3)

x_s = np.arange(len(TMC_Shap_AIDS_Pred))
x_1 = np.arange(len(TMC_train3))
plt.figure(figsize = (10, 6))

plt.stem(x_s, TMC_Shap_AIDS_Pred, linefmt = 'tab:blue', markerfmt = '', basefmt = '', label = 'Predicted Shapley Values')
plt.axhline(avg_s, color = 'tab:blue', linestyle = '--', linewidth = 2, label = 'Average Predicted Values')
plt.stem(x_1, TMC_train3, linefmt = 'tab:orange', markerfmt = '', basefmt = '', label = 'TMC Shapley Values')
plt.axhline(avg_1, color = 'tab:orange', linestyle = '--', linewidth = 2, label = 'Average TMC Shapley Values')

plt.xlabel('Training Data Index', fontsize = 14, weight = 'bold')
plt.ylabel('Data Shapley Values', fontsize = 14, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'TMC_Comparison_(10+1)_1.png'))

'Linear Plot'
true_vals = np.asarray(TMC_train3)
pred_vals = np.asarray(TMC_Shap_AIDS_Pred)
plt.figure(figsize = (8,8))
plt.scatter(true_vals, pred_vals, s = 30, color = 'tab:blue', alpha = 0.6, edgecolors = 'k',
            linewidth = 0.5, label = 'Predicted TMC Shapley Values vs True TMC Shapley Values')
all_min = min(true_vals.min(), pred_vals.min())
all_max = max(true_vals.max(), pred_vals.max())
plt.plot([all_min, all_max], [all_min, all_max], '--k', linewidth = 1.5, label = 'y=x')
plt.xlabel('True TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.ylabel('Predicted TMC Shapley Values', fontsize = 10, weight = 'bold')
plt.title('True vs Predicted TMC Shapley Values', fontsize = 12, weight = 'bold')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.legend(fontsize = 10)
out_path = os.path.join(GRAPH_DIR, 'TMC_LinearCompareAIDS(10+1)_1.png')
plt.tight_layout()
plt.savefig(out_path, dpi = 500)
plt.close()
print(f'Saved')

# File paths
files = [
    'Outputs/250619/Trial01/Iris_01_5pts_CPU',
    'Outputs/250619/Trial01/Iris_01_10pts_CPU',
    'Outputs/250619/Trial01/Iris_01_15pts_CPU',
    'Outputs/250619/Trial01/Iris_01_20pts_CPU',
]
labels = ['5 pts', '10 pts', '15 pts', '20 pts']

for file, label in zip(files, labels):
    df = pd.read_csv(file)
    vals = df['TrueDShap']
    avg = vals.mean()
    print(f'Average TrueDShap for {label}: {avg:.6f}')
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=10, alpha=0.7, color='tab:blue', edgecolor='black')
    plt.xlabel('TrueDShap')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of TrueDShap ({label})')
    plt.tight_layout()
    plt.show()