'''
0. Import Packages
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd

from torch_geometric.nn import TransformerConv, GINConv, global_add_pool
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU, Dropout

'''
1. Scaled TMC-Shapley Values by Utility
'''
def total_utility(X_train, y_train, X_test, y_test, baseline=0.5):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc - baseline

def utility_normalisation(phi, total_utility):
    return phi / (total_utility + 1e-20)

'''
2. Gaussian Affinity Matrix
'''
def compute_Affinity(X, k=15):
    D = pairwise_distances(X, metric='euclidean')
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    A   = np.zeros_like(D)
    row = np.repeat(np.arange(X.shape[0]), k)
    col = idx.flatten()
    A[row, col] = 1.0
    return A, None

'''
3. Randomized SVD
'''
def SVD_Transform(A, d=32):
    U, S, _ = randomized_svd(A, n_components=d, random_state=42)
    return U * np.sqrt(S)

'''
4. Graph Construction
'''
def build_graph(X, A, X_raw=None, y=None, tmc=None):
    ei = np.array(np.nonzero(A)).T
    edge_index = torch.tensor(ei,  dtype=torch.long).t().contiguous()
    edge_attr  = torch.ones(ei.shape[0], 1)                     # weight = 1

    data       = Data(x=torch.tensor(X, dtype=torch.float),
                      edge_index=edge_index,
                      edge_attr=edge_attr)
    data.X_raw, data.y_raw = X_raw, y
    if tmc is not None:
        data.y = torch.tensor(tmc, dtype=torch.float).unsqueeze(1)
    return data

'''
5. Model Construction
'''
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128,
                 out_channels=1, heads=4, dropout=0.1):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), ReLU(), Linear(o, o))
        self.conv1 = GINConv(mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(mlp(hidden_channels, hidden_channels))
        self.gt    = TransformerConv(hidden_channels,
                                     hidden_channels//heads,
                                     heads=heads, edge_dim=1)
        self.out   = Sequential(Linear(hidden_channels, hidden_channels),
                                Dropout(dropout),
                                Linear(hidden_channels, out_channels))
    def forward(self, d):
        x = F.relu(self.conv1(d.x, d.edge_index))
        x = self.conv2(x, d.edge_index)
        x = self.gt(x,  d.edge_index, d.edge_attr)
        return self.out(x)


'''
6. Data Preprocessing
'''
def load_and_preprocess(csv_path, X_test, y_test, d=32):
    df    = pd.read_csv(csv_path)
    X_raw = pd.get_dummies(df.drop(columns=['label','TMC_Shap','G_Shap','LOO'])).to_numpy()
    y     = df['label'].to_numpy()
    phi   = df['TMC_Shap'].to_numpy()

    util      = total_utility(X_raw, y, X_test, y_test)
    phi_norm  = utility_normalisation(phi, util)

    phi_norm -= phi_norm.mean()

    A     = compute_Affinity(X_raw)[0]
    X_svd = SVD_Transform(A, d=d)
    return build_graph(X_svd, A, X_raw=X_raw, y=y, tmc=phi_norm)

'''
7. Training
'''
def train(model, data_list, X_test, y_test,
          epochs=300, lr=1e-3, B=32,
          lam_eff=1.0, lam_var=0.1, lam_sign=2.0):
    """
    model      – GNN to be optimised
    data_list  – list of torch_geometric.data.Data objects (each with .X_raw, .y_raw, .y)
    X_test     – test-set features  (NumPy array)
    y_test     – test-set labels    (NumPy array)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        total_loss_ep = 0.0
        optimizer.zero_grad()

        for data in data_list:
            X_raw, y_raw = data.X_raw, data.y_raw
            phi_hat = model(data).view(-1)                # predicted Shapley values
            N       = phi_hat.numel()

            'Loss on Monte Carlo Estimation'
            mc_losses = []
            for _ in range(B):
                i      = np.random.randint(N)
                mask   = np.ones(N, dtype=bool); mask[i] = False
                S_idx  = np.where(mask & (np.random.rand(N) < 0.5))[0]

                v_S   = total_utility(X_raw[S_idx], y_raw[S_idx], X_test, y_test)
                v_Si  = total_utility(np.vstack([X_raw[S_idx], X_raw[i]]),
                                     np.hstack([y_raw[S_idx], y_raw[i]]),
                                     X_test, y_test)
                mc_losses.append((phi_hat[i] - (v_Si - v_S))**2)

            loss_mc = torch.stack(mc_losses).mean()

            'Efficiency penalty term'
            phi_true = data.y.view(-1)
            loss_eff = ((phi_hat.sum() - phi_true.sum()) / N) ** 2
            loss_var = (phi_hat.var(unbiased=False) -
                        phi_true.var(unbiased=False)) ** 2

            'Sign penalty term'
            sign_target = (phi_true > 0).float()
            loss_sign   = F.binary_cross_entropy_with_logits(phi_hat, sign_target)

            'Total Loss'
            loss = (loss_mc
                    + lam_eff  * loss_eff
                    + lam_var  * loss_var
                    + lam_sign * loss_sign)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss_ep += loss.item()

        optimizer.step()

        if ep % 50 == 0:
            print(f'Epoch {ep:3d} | total loss {total_loss_ep:.4f}')

'''
8. Prediction
'''
@torch.no_grad()
def predict(model, data): return model(data).squeeze().cpu().numpy()

'''
9. Implementation
'''
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
    BASE2 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515'
    paths  = [f'{BASE}/train{i}.csv' for i in (1,2,3)]

    test_df = pd.read_csv(f'{BASE}/test.csv')
    X_test  = pd.get_dummies(test_df.drop(columns=['label'])).to_numpy()
    y_test  = test_df['label'].to_numpy()

    data1 = load_and_preprocess(paths[0], X_test, y_test)
    data2 = load_and_preprocess(paths[1], X_test, y_test)
    data3 = load_and_preprocess(paths[2], X_test, y_test)

    gnn = GINShap(data1.num_node_features, hidden_channels=256)

    train(gnn, [data1, data2], X_test, y_test)

    preds = predict(gnn, data3)           
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df3 = pd.read_csv(paths[2])
    df3['predicted_TMC_Shap'] = preds
    out_csv = f'{BASE2}/train3_predicted5.csv'
    df3.to_csv(out_csv, index=False)
    print('saved ➜', out_csv)

'''
10. Performance Plots
'''
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
DATA_DIR   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
TEST_CSV   = os.path.join(DATA_DIR, 'test.csv')
PRED_CSV   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS/train3_predicted5.csv'
df3_pred   = pd.read_csv(PRED_CSV)
TMC_true   = df3_pred['TMC_Shap'].values
LOO_vals   = df3_pred['LOO'].values
G_true     = df3_pred['G_Shap'].values
pred       = df3_pred['predicted_TMC_Shap']

X_train3 = pd.get_dummies(
    df3_pred.drop(columns=['label','G_Shap','LOO','TMC_Shap','predicted_TMC_Shap'])
).to_numpy()
y_train3 = df3_pred['label'].to_numpy()

df_test = pd.read_csv(TEST_CSV)
X_test  = pd.get_dummies(df_test.drop(columns=['label'])).to_numpy()
y_test  = df_test['label'].to_numpy()

PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515', 'AIDS3pred5_Low.png')
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_true,
          'predicted TMC': pred,
          'G Shapley': G_true,
          'LOO': LOO_vals},
    X_train=X_train3, y_train=y_train3,
    X_test=X_test, y_test=y_test,
    a=0.05, b=0.4, ylim=(72,85),
    order='ascending', random_iters=10,
    save_path=PERF_PATH
)