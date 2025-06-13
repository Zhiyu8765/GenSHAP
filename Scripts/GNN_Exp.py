'''
0. Import Packages
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchsort
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
1. Gaussian Affinity Matrix
'''
def compute_Affinity(X):
    D = pairwise_distances(X, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2)/(2 * (sigma ** 2)))
    A = np.maximum(A, A.T) # ensure symmetry
    return A, sigma


'''
2. Randomized SVD
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
    U, S, _ = randomized_svd(A, n_components=d, random_state=42)
    X_svd = U * np.sqrt(S)
    X_t = torch.tensor(X_svd, dtype=torch.float)  # ← mod
    X_t = filp_pe_sign(X_t, method='l1')  # ← mod
    X_svd = X_t.numpy()
    return X_svd


'''
3. Graph Construction
'''
def build_graph(X_raw, d = 10, tmc = None):
    A, sigma = compute_Affinity(X_raw)
    X_svd = SVD_Transform(A, d = d)

    N = A.shape[0]
    upper_tri = np.triu_indices(N, k = 1)
    src = upper_tri[0]
    dst = upper_tri[1]

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = A[src, dst]
    edge_attr = torch.tensor(edge_weight, dtype = torch.float).unsqueeze(1)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    data = Data(x = torch.tensor(X_svd, dtype = torch.float),
                edge_index = edge_index, edge_attr = edge_attr)
    data.X_raw = torch.tensor(X_raw, dtype = torch.float)
    if tmc is not None:
        data.tmc = torch.tensor(tmc, dtype = torch.float).unsqueeze(1)

    return data


'''
4. Model Construction
'''
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=4096,
                 out_channels=1, heads=64, dropout=0.3):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))

        # GIN
        self.conv1 = GINConv(mlp(in_channels, hidden_channels))
        # H
        self.H = Sequential(Linear(hidden_channels, hidden_channels), LeakyReLU())
        # GT
        self.gt    = TransformerConv(hidden_channels,
                                     hidden_channels//heads,
                                     heads=heads, edge_dim=1)
        # MLP
        self.out   = Sequential(Linear(hidden_channels, hidden_channels),
                                Dropout(dropout),
                                Linear(hidden_channels, out_channels))

    def forward(self, d):
        x = self.conv1(d.x, d.edge_index)
        x = self.H(x)
        x = self.gt(x, d.edge_index, d.edge_attr)
        return self.out(x)


# ### Deeper GIN
# class GINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=512,
#                  out_channels=1, heads=8, dropout=0.2):
#         super().__init__()
#
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))
#
#         self.conv1 = GINConv(mlp(in_channels, hidden_channels))
#         self.conv2 = GINConv(mlp(hidden_channels, hidden_channels))
#         self.H = Sequential(Linear(hidden_channels, hidden_channels), LeakyReLU())
#         self.gt = TransformerConv(hidden_channels, hidden_channels//heads, heads = heads, edge_dim = 1)
#         self.out = Sequential(Linear(hidden_channels, hidden_channels), Dropout(dropout),
#                                 Linear(hidden_channels, out_channels))
#
#     def forward(self, d):
#         x = self.conv1(d.x, d.edge_index)
#         x = self.conv2(x, d.edge_index)
#         x = self.H(x)
#         x = self.gt(x, d.edge_index, d.edge_attr)
#         return self.out(x)



'''
6. Data Preprocessing
'''
def load_and_preprocess(csv_path, d=10):
    df = pd.read_csv(csv_path)
    X_raw = pd.get_dummies(df.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
    phi = df['TrueDShap'].to_numpy()
    scaler = MinMaxScaler()
    phi_scaled = scaler.fit_transform(phi.reshape(-1, 1)).flatten()
    A     = compute_Affinity(X_raw)[0]
    return build_graph(X_raw = X_raw, d = 10, tmc = phi_scaled), scaler

'''
7. Training
'''
# def shapley_loss_with_sign_and_rank(phi_hat, phi_true, lam_sign=0.5):
#     mse = F.mse_loss(phi_hat, phi_true)
#     sign_penalty = torch.relu(-phi_hat * phi_true).mean()
#     return mse + lam_sign * sign_penalty

def train(model, data_list, epochs=500, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    for ep in range(1, epochs + 1):
        total_loss_ep = 0.0
        optimizer.zero_grad()
        per_node_losses = []

        for data in data_list:
            phi_hat = model(data).view(-1)
            phi_true = data.tmc.view(-1)
            #loss_i = F.l1_loss(phi_hat, phi_true, reduction='mean')
            loss_i = F.mse_loss(phi_hat, phi_true, reduction='mean')
            #loss_i = shapley_loss_with_sign_and_rank(phi_hat, phi_true, lam_sign=lam_sign)
            per_node_losses.append(loss_i)
            total_loss_ep += loss_i.item()

        epoch_loss = torch.stack(per_node_losses).sum()
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 5 == 0:
            print(f'Epoch {ep:3d} | total loss {total_loss_ep:.4f}| LR: {scheduler.get_last_lr()[0]:.2e}')

'''
8. Prediction
'''
def predict(model, data):
    with torch.no_grad():
        return model(data).squeeze().cpu().numpy()



######################################################################################################
######################################################################################################
###############################  Experiments #########################################################
'''Iris'''
'A. Implementation'
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612'
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
        load_and_preprocess(train_csv)[0]
        for train_csv in Train_list
    ]
    data_holdout, scaler = load_and_preprocess(holdout_train_csv)

    # all_phi_scaled = torch.cat([d.tmc.view(-1) for d in data_list], dim = 0)
    # bias_init = float(all_phi_scaled.mean().item())

    gnn = GINShap(in_channels=10, hidden_channels=4096)
    train(gnn, data_list, epochs=200, lr=1e-4)

    preds = predict(gnn, data_holdout)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/Iris_12_Pred6.csv'
    df_holdout.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

'Removal Plots'
df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TestIris_12')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Iris_12_TMC')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','TrueDShap','TMC_Shap'])).to_numpy()
y_train = df_train['target'].to_numpy()
True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_train['TMC_Shap'].to_numpy()
Pred_Shap = preds
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612', 'Iris12Pred_High6.png')
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
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250612'
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

# grouped bars
plt.bar(indices - width, true_vals,  width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
plt.bar(indices, tmc_vals, width, label='TMC Shapley', color='tab:orange',edgecolor='k', alpha=0.7)
plt.bar(indices + width, pred_vals, width, label='Predicted Shapley',  color='tab:green', edgecolor='k', alpha=0.7)

# horizontal lines at each mean
plt.plot(indices, [mean_true]*N, linestyle='--', linewidth=2, label=f'Average True Shapley Values ({mean_true:.3f})',   color='tab:blue')
plt.plot(indices, [mean_tmc]*N, linestyle='--', linewidth=2, label=f'Average TMC Shapley Values ({mean_tmc:.3f})',     color='tab:orange')
plt.plot(indices, [mean_pred]*N, linestyle='--', linewidth=2, label=f'Average Predicted Shapley Values ({mean_pred:.3f})',color='tab:green')

# labels & legend
plt.xlabel('Sample Index', fontsize=12, weight='bold')
plt.ylabel('Shapley Value', fontsize=12, weight='bold')
plt.title('Comparison of True, TMC & Predicted Shapley Values', fontsize=14, weight='bold')
plt.xticks(indices)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, 'Shapley_compare6.png')
plt.savefig(out_path, dpi=500)

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

out_path = os.path.join(GRAPH_DIR, 'TMC_LinearCompare6.png')
plt.tight_layout()
plt.savefig(out_path, dpi=500)



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