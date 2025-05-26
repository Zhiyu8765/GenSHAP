'''
0. Import Packages
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchsort

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
    return phi / (total_utility)

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
def load_and_preprocess(csv_path,test_csv_path, d=32):
    df    = pd.read_csv(csv_path)
    X_raw = pd.get_dummies(df.drop(columns=['label','TMC_Shap','G_Shap','LOO'])).to_numpy()
    y     = df['label'].to_numpy()
    phi   = df['TMC_Shap'].to_numpy()

    df_test = pd.read_csv(test_csv_path)
    X_test = pd.get_dummies(df_test.drop(columns = ['label'])).to_numpy()
    y_test = df_test['label'].to_numpy()

    util      = total_utility(X_raw, y, X_test, y_test)
    phi_norm  = utility_normalisation(phi, util)

    A     = compute_Affinity(X_raw)[0]
    X_svd = SVD_Transform(A, d=d)
    return build_graph(X_svd, A, X_raw=X_raw, y=y, tmc=phi_norm)

'''
7. Training
'''
def train(model, data_list, epochs=500, lr=1e-4, lam_sign=1.0, lam_eff = 0.5, lam_rank = 1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        total_loss_ep = 0.0
        optimizer.zero_grad()

        for data in data_list:
            phi_hat = model(data).view(-1)
            phi_true = data.y.view(-1)
            N = phi_hat.shape[0]

            '1. Relative MSE'
            relative_MSE = ((phi_hat - phi_true)/(phi_true).abs() + 1e-10 ).pow(2).mean()

            '2. Penalty term on sign'
            loss_sign = torch.relu(-phi_hat * phi_true).mean()

            '3. Penalty on efficiency property'
            loss_eff = (phi_hat.sum() - phi_true.sum()).pow(2)

            '4. Penalty on ranking'
            TAU = 1.0
            rank_pred = torchsort.soft_rank(
                phi_hat.unsqueeze(0),
                regularization_strength=TAU
            ).squeeze(0)
            rank_true = torchsort.soft_rank(
                phi_true.unsqueeze(0),
                regularization_strength=TAU
            ).squeeze(0)
            loss_rank = F.mse_loss(rank_pred, rank_true)

            'Total Loss'
            loss = (relative_MSE + lam_sign * loss_sign + lam_eff * loss_eff +  lam_rank * loss_rank)
            #loss = loss/len(data_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_loss_ep += loss.item()

        optimizer.step()

        if ep % 50 == 0:
            print(f'Epoch {ep:3d} | total loss {total_loss_ep:.4f}')

'''
8. Prediction
'''
def predict(model, data):
    with torch.no_grad():
        return model(data).squeeze().cpu().numpy()

'''
9. Implementation
'''
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/AIDS'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515'
    Train_Test_Pairs = [
        (f'{BASE}/train1.csv', f'{BASE}/test.csv'),
        (f'{BASE}/train2.csv', f'{BASE}/test.csv'),
        (f'{BASE}/train4.csv', f'{BASE}/test4.csv'),
        (f'{BASE}/train5.csv', f'{BASE}/test5.csv'),
        (f'{BASE}/train6.csv', f'{BASE}/test6.csv')
    ]
    holdout_train_csv = f'{BASE}/train3.csv'
    holdout_test_csv = f'{BASE}/test.csv'

    data_list = [
        load_and_preprocess(train_csv, test_csv)
        for train_csv, test_csv in Train_Test_Pairs
    ]
    data_holdout = load_and_preprocess(holdout_train_csv, holdout_test_csv)


    gnn = GINShap(data_list[0].num_node_features, hidden_channels=256)
    train(gnn, data_list, epochs=700, lr=1e-2, lam_sign=0.25, lam_eff=0.5, lam_rank=10.0)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/train3_predicted(5+1).csv'
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
LOO_vals = df_holdout['LOO'].to_numpy()
G_true = df_holdout['G_Shap'].to_numpy()

PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515', 'AIDS3pred_(5+1)_high.png')
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



'''
9. Implementation: MNIST
'''
if __name__ == '__main__':
    BASE   = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/MNIST'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515'
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


    gnn = GINShap(data_list[0].num_node_features, hidden_channels=256)
    train(gnn, data_list, epochs=500, lr=1e-3, lam_sign=0.5, lam_eff=0.5)

    preds = predict(gnn, data_holdout)
    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['predicted_TMC_Shap'] = preds
    out_csv = f'{PATH_OUT}/trainMNIST45_predicted(4+1).csv'
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
LOO_vals = df_holdout['LOO'].to_numpy()
G_true = df_holdout['G_Shap'].to_numpy()

PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250515', 'MNISTpred_(4+1)_Low.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'TMC Shapley': TMC_true,
          'predicted TMC': preds,
          'G Shapley': G_true,
          'LOO': LOO_vals},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (47,60),
    order = 'ascending', random_iters = 10,
    save_path = PERF_PATH
)