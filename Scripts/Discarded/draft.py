############################################################
# 00  Imports
############################################################
import os, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU, Dropout

torch.manual_seed(42)
np.random.seed(42)

############################################################
# 01  Utility function  (no clamping → can be negative)
############################################################
def total_utility(X_tr, y_tr, X_te, y_te, baseline=0.5):
    mdl = LogisticRegression(max_iter=1000)
    mdl.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mdl.predict(X_te))
    return acc - baseline          # may be < 0

def normalise_phi(phi, util):
    return phi / (util + 1e-12)

############################################################
# 02  k-NN affinity matrix  (avoids over-smoothing)
############################################################
def knn_affinity(X, k=15):
    D = pairwise_distances(X, metric='euclidean')
    idx = np.argsort(D, axis=1)[:, 1:k+1]         # k nearest (skip self)
    A   = np.zeros_like(D, dtype=np.float32)
    rows = np.repeat(np.arange(X.shape[0]), k)
    A[rows, idx.ravel()] = 1.0
    return A

############################################################
# 03  Graph embedding via SVD
############################################################
def svd_embed(A, d=32):
    U, S, _ = randomized_svd(A, n_components=d, random_state=42)
    return U * np.sqrt(S)

############################################################
# 04  Build torch-geometric Data object
############################################################
def build_graph(X_emb, A, X_raw, y_raw, phi):
    ei = np.array(np.nonzero(A)).T
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    edge_attr  = torch.ones(ei.shape[0], 1)   # unit weights

    data       = Data(x=torch.tensor(X_emb, dtype=torch.float32),
                      edge_index=edge_index,
                      edge_attr=edge_attr)
    data.X_raw, data.y_raw = X_raw, y_raw
    data.phi_true = torch.tensor(phi, dtype=torch.float32).unsqueeze(1)
    return data

############################################################
# 05  Simple GIN + TransformerConv predictor
############################################################
class GINShap(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        def mlp(i, o): return Sequential(Linear(i, o), ReLU(), Linear(o, o))
        self.conv1 = GINConv(mlp(in_dim, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.gt    = TransformerConv(hidden, hidden // heads,
                                     heads=heads, edge_dim=1)
        self.head  = Sequential(Linear(hidden, hidden),
                                Dropout(dropout),
                                Linear(hidden, 1))

    def forward(self, d):
        x = torch.relu(self.conv1(d.x, d.edge_index))
        x = self.conv2(x, d.edge_index)
        x = self.gt(x, d.edge_index, d.edge_attr)
        return self.head(x)

############################################################
# 06  CSV → Data
############################################################
def load_csv(file_path, X_test, y_test, embed_dim=32):
    df      = pd.read_csv(file_path)
    X_raw   = pd.get_dummies(df.drop(columns=['label',
                                              'TMC_Shap',
                                              'G_Shap',
                                              'LOO'])).to_numpy()
    y_raw   = df['label'].to_numpy()
    phi     = df['TMC_Shap'].to_numpy()

    util    = total_utility(X_raw, y_raw, X_test, y_test)
    phi_n   = normalise_phi(phi, util)
    phi_n  -= phi_n.mean()                      # centre → negatives exist

    A       = knn_affinity(X_raw, k=15)
    X_emb   = svd_embed(A, d=embed_dim)
    return build_graph(X_emb, A, X_raw, y_raw, phi_n)

############################################################
# 07  Training loop  (compact loss)
############################################################
def train(model, data_list, X_test, y_test,
          epochs=300, B=32, lr=1e-3,
          lambda_mean=1.0, lambda_sign=1.0, margin=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        total_loss_ep = 0.0
        opt.zero_grad()

        for data in data_list:
            X_raw, y_raw = data.X_raw, data.y_raw
            phi_true = data.phi_true.view(-1)
            phi_hat  = model(data).view(-1)
            N = phi_hat.numel()

            # L_MC
            mc_losses = []
            for _ in range(B):
                idx_i = np.random.randint(N)
                mask  = np.ones(N, dtype=bool); mask[idx_i] = False
                S_idx = np.where(mask & (np.random.rand(N) < 0.5))[0]
                v_S  = total_utility(X_raw[S_idx], y_raw[S_idx],
                                     X_test, y_test)
                v_Si = total_utility(np.vstack([X_raw[S_idx], X_raw[idx_i]]),
                                     np.hstack([y_raw[S_idx], y_raw[idx_i]]),
                                     X_test, y_test)
                mc_losses.append((phi_hat[idx_i] - (v_Si - v_S))**2)
            loss_mc = torch.stack(mc_losses).mean()

            # L_mean
            loss_mean = (phi_hat.mean() - phi_true.mean())**2

            # L_sign  (hinge with small margin)
            sign_target = torch.sign(phi_true)
            loss_sign = torch.relu(margin - sign_target * phi_hat).mean()

            loss = loss_mc + lambda_mean*loss_mean + lambda_sign*loss_sign
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_loss_ep += loss.item()

        opt.step()
        if ep % 50 == 0:
            print(f'Epoch {ep:3d} | total loss {total_loss_ep:.4f}')

############################################################
# 08  Prediction helper
############################################################
@torch.no_grad()
def predict(model, data):
    return model(data).squeeze().cpu().numpy()

############################################################
# 09  Main
############################################################
if __name__ == '__main__':
    BASE = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
    paths = [f'{BASE}/train{i}.csv' for i in (1, 2, 3)]

    # test set
    df_test = pd.read_csv(f'{BASE}/test.csv')
    X_test  = pd.get_dummies(df_test.drop(columns=['label'])).to_numpy()
    y_test  = df_test['label'].to_numpy()

    # load data
    data1 = load_csv(paths[0], X_test, y_test)
    data2 = load_csv(paths[1], X_test, y_test)
    data3 = load_csv(paths[2], X_test, y_test)

    # model
    gnn = GINShap(in_dim=data1.num_node_features, hidden=256)

    # train on train1 + train2
    train(gnn, [data1, data2], X_test, y_test)

    # predict on train3
    preds = predict(gnn, data3)
    print('prediction percentiles:', np.percentile(preds, [0, 5, 50, 95, 100]))

    # save CSV
    df3 = pd.read_csv(paths[2])
    df3['predicted_TMC_Shap'] = preds
    out_csv = f'{BASE}/train3_predicted_compact.csv'
    df3.to_csv(out_csv, index=False)
    print('saved →', out_csv)

    # performance plot (unchanged)
    from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
    dfp  = pd.read_csv(out_csv)
    vals = {'TMC Shapley'  : dfp['TMC_Shap'].values,
            'predicted TMC': dfp['predicted_TMC_Shap'].values,
            'G Shapley'    : dfp['G_Shap'].values,
            'LOO'          : dfp['LOO'].values}

    X_train3 = pd.get_dummies(
        dfp.drop(columns=['label', 'G_Shap', 'LOO',
                          'TMC_Shap', 'predicted_TMC_Shap'])
    ).to_numpy()
    y_train3 = dfp['label'].to_numpy()

    out_plot = os.path.join(BASE, 'AIDS3pred_compact_low.png')
    removal_percentages, results = PerformancePlot_Removal(
        vals={'TMC Shapley': dfp['TMC_Shap'].values,
              'predicted TMC': dfp['predicted_TMC_Shap'].values,
              'G Shapley': dfp['G_Shap'].values,
              'LOO': dfp['LOO'].values},
        X_train=X_train3, y_train=y_train3,
        X_test=X_test, y_test=y_test,
        a=0.05, b=0.4, ylim=(72, 85),
        order='ascending', random_iters=10,
        save_path=out_plot
    )


out_plot = os.path.join(BASE, 'AIDS3pred_compact_High.png')
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': dfp['TMC_Shap'].values,
          'predicted TMC': dfp['predicted_TMC_Shap'].values,
          'G Shapley': dfp['G_Shap'].values,
          'LOO': dfp['LOO'].values},
    X_train=X_train3, y_train=y_train3,
    X_test=X_test, y_test=y_test,
    a=0.05, b=0.4, ylim=(40, 85),
    order='descending', random_iters=10,
    save_path=out_plot
)