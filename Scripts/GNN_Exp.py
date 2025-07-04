'''
0. Import Packages
'''
import os
import numpy as np
import pandas as pd
import torch
import torchsort
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl
from dgl.nn.pytorch.conv import GINConv
from mpmath.calculus.extrapolation import standardize
from torch_geometric.data import Data
from torch.nn import (Sequential, Linear, LeakyReLU, Dropout,
                      TransformerEncoder, TransformerEncoderLayer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.extmath import randomized_svd
from tensorflow.python.ops.gen_nn_ops import LeakyRelu
device = torch.device("cpu")


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
    X_t = filp_pe_sign(X_t, method='sum') # l0 (positive) / sum
    return X_t.numpy()


'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, phi, total_util, c, d = 10):
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    X_svd = SVD_Transform(A, d = d)

    N = A.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(np.concatenate([eidx, eidx[::-1]], axis = 1), dtype = torch.long)
    w = np.concatenate([A[src, dst], A[src, dst]])
    edge_attr = torch.tensor(w, dtype = torch.float).unsqueeze(1)

    data = Data(
        x = torch.tensor(X_svd, dtype = torch.float),
        edge_index = edge_index,
        edge_attr = edge_attr
    )
    data.X_raw = torch.tensor(X_raw, dtype = torch.float)
    data.tmc = torch.tensor(phi, dtype = torch.float)
    data.total_util = torch.tensor(total_util, dtype = torch.float)

    return data


'''
3. Data Preprocessing
'''
def load_and_preprocess(train_csv, test_csv, d=10):
    df_train = pd.read_csv(train_csv)
    X_raw = pd.get_dummies(df_train.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
    phi = df_train['TrueDShap'].to_numpy()
    df_test = pd.read_csv(test_csv)
    y_test = df_test['target'].to_numpy()
    maj = pd.Series(y_test).mode().iloc[0]
    baseline = (y_test == maj).astype(float)
    total_util = baseline.mean()

    data_list = []
    for c in C_List:
        data = build_graph(X_raw, phi, total_util, d = d, c = c)
        data_list.append(data)

    return data_list


'''
4. Model Construction
'''
''' Version A. Without BatchNorm '''
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels = 64, out_channels = 1, heads = 2):
        super().__init__()
        def mlp(i, o):
            return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))

        # 5 Parallel GIN Branches
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(3):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                            aggregator_type = 'sum', init_eps = 0.1, learn_eps = True)
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))

        # GT (via TransformerEncoder)
        d_concat = hidden_channels * len(C_List) # 64 or 128 * 5 = 640
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.1, batch_first = True)
        self.gt = TransformerEncoder(encoder_layer, num_layers = 3)

        # Z
        self.Z = Sequential(Linear(d_concat, hidden_channels), LeakyReLU())

        # MLP (widen by hidden_channels * 2)
        self.out = Sequential(
            Linear(hidden_channels + 2, hidden_channels * 2),
            LeakyReLU(),
            Linear(hidden_channels * 2, hidden_channels * 2),
            LeakyReLU(),
            Linear(hidden_channels * 2, out_channels)
        )

    def forward(self, data_list):
        data = data_list[0].to(device)
        x0 = data.x # Extract nodes
        ei = data.edge_index # Extract edge indices
        ew = data.edge_attr.view(-1) # Extract edge weights for DGL

        # DGL Graph
        g = dgl.graph((ei[0], ei[1]), num_nodes = x0.size(0))
        g.edata['w'] = ew

        # Parallel GIN
        output_GIN = []
        for convs, gdata in zip(self.branches, data_list):
            node_input = gdata.x.to(device)
            for conv in convs:
                node_input = conv(g, node_input , edge_weight = g.edata['w'])
            output_GIN.append(node_input)
        output_GIN = torch.cat(output_GIN, dim = 1)

        # Skip-Connection
        output_skip = output_GIN + self.skip_lin(x0)

        # GT
        output_skip = output_skip.unsqueeze(0) # transform to: (1, N, hidden_channels*5)
        output_gt = self.gt(output_skip)
        output_gt = output_gt.squeeze(0)

        # H
        output_h = self.Z(output_gt)

        # total utility + (N/1000) -> MLP
        N = output_h.size(0)
        total_util = data.total_util
        utility_vec = total_util.expand(N, 1)
        NU_vector = torch.full((N, 1), np.sqrt(N)/10., device = device)
        output_final = torch.cat([output_h, utility_vec, NU_vector], dim = 1)

        return self.out(output_final)


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

def train(model, dataset_graphs, epochs=200, lr=1e-4, alpha = 0.2):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # adamw
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs) # cosine
    loss_history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for data_list in dataset_graphs:
            phi_hat = model(data_list).view(-1)
            phi_true = data_list[0].tmc.to(device)

            mean_true = phi_true.mean()
            std_true = phi_true.std(unbiased=False) + 1e-10
            phi_true_scaled = (phi_true - mean_true) / std_true

            mean_hat = phi_hat.mean()
            std_hat = phi_hat.std(unbiased=False) + 1e-10
            phi_hat_scaled = (phi_hat - mean_hat) / std_hat

            loss_MSE = F.mse_loss(phi_hat, phi_true, reduction = 'mean')
            #loss_Rank = BPR_loss(phi_hat_scaled, phi_true_scaled)
            #loss = (1-alpha) * loss_MSE + alpha * loss_Rank
            loss = loss_MSE
            losses.append(loss)
        L = torch.stack(losses).mean()
        loss_history.append(L.item())

        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        #scheduler.step()

        if ep % 5 == 0:
            #current_lr = scheduler.get_last_lr()[0]
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {ep:3d} | total loss {L:.8f}| LR: {current_lr:.2e}')

    return loss_history


'''
6. Prediction
'''
def predict(model, data_list):
    model.eval()
    d = data_list[0].to(device)
    with torch.no_grad():
        return model(data_list).squeeze().cpu().numpy()


######################################################################################################
######################################################################################################
#########################################  Experiments ###############################################

'''
EXP 1. Iris (Subsets of label 1 & 2 for testing)
'''

'A. Implementation'
if __name__ == '__main__':
    BASE00 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2'
    BASE01 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group01'
    BASE02 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group03'
    BASE03 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04'
    BASEBC = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group05'
    #BASE = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V2'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250703/Trial09'
    #PATH_OUT = '/tmp/pycharm_project_236/Outputs/250623/Trial01'
    # Train_list = [
    #     (f'{BASE01}/Iris_01_1'),(f'{BASE01}/Iris_01_2'),
    #     (f'{BASE01}/Iris_01_3'),(f'{BASE01}/Iris_01_4'),
    #     (f'{BASE01}/Iris_01_5'),(f'{BASE01}/Iris_01_6'),
    #     (f'{BASE01}/Iris_01_7'),(f'{BASE01}/Iris_01_8'),
    #     (f'{BASE01}/Iris_01_9'),(f'{BASE01}/Iris_01_10'),
    #     (f'{BASE01}/Iris_01_11'),(f'{BASE01}/Iris_01_12'),
    #     (f'{BASE01}/Iris_01_13'),(f'{BASE01}/Iris_01_14'),
    #     (f'{BASE01}/Iris_01_15'),(f'{BASE01}/Iris_01_16'),
    #     (f'{BASE01}/Iris_01_17'), (f'{BASE01}/Iris_01_18'),
    #     (f'{BASE01}/Iris_01_19'), (f'{BASE01}/Iris_01_20'),
    #     (f'{BASE01}/Iris_01_21'), (f'{BASE01}/Iris_01_22'),
    #     (f'{BASE01}/Iris_01_23'), (f'{BASE01}/Iris_01_24'),
    #     (f'{BASE01}/Iris_01_25'), (f'{BASE01}/Iris_01_26'),
    #     (f'{BASE01}/Iris_01_27'), (f'{BASE01}/Iris_01_28'),
    #     (f'{BASE01}/Iris_01_29'), (f'{BASE01}/Iris_01_30'),
    #     (f'{BASE02}/Iris_02_1'), (f'{BASE02}/Iris_02_2'),
    #     (f'{BASE02}/Iris_02_3'), (f'{BASE02}/Iris_02_4'),
    #     (f'{BASE02}/Iris_02_5'), (f'{BASE02}/Iris_02_6'),
    #     (f'{BASE02}/Iris_02_7'), (f'{BASE02}/Iris_02_8'),
    #     (f'{BASE02}/Iris_02_9'), (f'{BASE02}/Iris_02_10'),
    #     (f'{BASE02}/Iris_02_11'), (f'{BASE02}/Iris_02_12'),
    #     (f'{BASE02}/Iris_02_13'), (f'{BASE02}/Iris_02_14'),
    #     (f'{BASE02}/Iris_02_15'), (f'{BASE02}/Iris_02_16'),
    #     (f'{BASE02}/Iris_02_17'), (f'{BASE02}/Iris_02_18'),
    #     (f'{BASE02}/Iris_02_19'), (f'{BASE02}/Iris_02_20')
    # ]
    # Train_list = [
    #     (f'{BASE00}/Iris_01'),
    #     (f'{BASE00}/Iris_02'),
    #     (f'{BASE00}/bc_1'),
    #     (f'{BASE00}/bc_2'),
    #     (f'{BASE00}/bc_3'),
    #     (f'{BASE00}/MNIST_01'),
    #     (f'{BASE00}/MNIST_12'),
    #     (f'{BASE00}/MNIST_23'),
    #     (f'{BASE00}/MNIST_34'),
    #     (f'{BASE00}/MNIST_45'),
    #     (f'{BASE00}/MNIST_56'),
    #     (f'{BASE00}/MNIST_67')
    # ]
    Train_list = [
        (f'{BASEBC}/bc_1.csv'), (f'{BASEBC}/bc_2.csv'),
        (f'{BASEBC}/bc_3.csv'), (f'{BASEBC}/bc_4.csv'),
        (f'{BASEBC}/bc_5.csv'), (f'{BASEBC}/bc_6.csv'),
        (f'{BASEBC}/bc_7.csv'), (f'{BASEBC}/bc_8.csv'),
        (f'{BASEBC}/bc_9.csv'), (f'{BASEBC}/bc_10.csv'),
        (f'{BASEBC}/bc_11.csv'), (f'{BASEBC}/bc_12.csv'),
        (f'{BASEBC}/bc_13.csv'), (f'{BASEBC}/bc_14.csv'),
        (f'{BASEBC}/bc_15.csv'), (f'{BASEBC}/bc_16.csv'),
        (f'{BASEBC}/bc_17.csv'), (f'{BASEBC}/bc_18.csv'),
        (f'{BASEBC}/bc_19.csv'), (f'{BASEBC}/bc_20.csv'),
        (f'{BASEBC}/bc_21.csv'), (f'{BASEBC}/bc_22.csv'),
        (f'{BASEBC}/bc_23.csv'), (f'{BASEBC}/bc_24.csv'),
        (f'{BASEBC}/bc_25.csv'), (f'{BASEBC}/bc_26.csv'),
        (f'{BASEBC}/bc_27.csv'), (f'{BASEBC}/bc_28.csv'),
        (f'{BASEBC}/bc_29.csv'), (f'{BASEBC}/bc_30.csv')
    ]

    # Test_list = [
    #     (f'{BASE01}/Iris_01_test_1'),(f'{BASE01}/Iris_01_test_2'),
    #     (f'{BASE01}/Iris_01_test_3'),(f'{BASE01}/Iris_01_test_4'),
    #     (f'{BASE01}/Iris_01_test_5'),(f'{BASE01}/Iris_01_test_6'),
    #     (f'{BASE01}/Iris_01_test_7'),(f'{BASE01}/Iris_01_test_8'),
    #     (f'{BASE01}/Iris_01_test_9'),(f'{BASE01}/Iris_01_test_10'),
    #     (f'{BASE01}/Iris_01_test_11'),(f'{BASE01}/Iris_01_test_12'),
    #     (f'{BASE01}/Iris_01_test_13'),(f'{BASE01}/Iris_01_test_14'),
    #     (f'{BASE01}/Iris_01_test_15'),(f'{BASE01}/Iris_01_test_16'),
    #     (f'{BASE01}/Iris_01_test_17'), (f'{BASE01}/Iris_01_test_18'),
    #     (f'{BASE01}/Iris_01_test_19'), (f'{BASE01}/Iris_01_test_20'),
    #     (f'{BASE01}/Iris_01_test_21'), (f'{BASE01}/Iris_01_test_22'),
    #     (f'{BASE01}/Iris_01_test_23'), (f'{BASE01}/Iris_01_test_24'),
    #     (f'{BASE01}/Iris_01_test_25'), (f'{BASE01}/Iris_01_test_26'),
    #     (f'{BASE01}/Iris_01_test_27'), (f'{BASE01}/Iris_01_test_28'),
    #     (f'{BASE01}/Iris_01_test_29'), (f'{BASE01}/Iris_01_test_30'),
    #     (f'{BASE02}/Iris_02_test_1'), (f'{BASE02}/Iris_02_test_2'),
    #     (f'{BASE02}/Iris_02_test_3'), (f'{BASE02}/Iris_02_test_4'),
    #     (f'{BASE02}/Iris_02_test_5'), (f'{BASE02}/Iris_02_test_6'),
    #     (f'{BASE02}/Iris_02_test_7'), (f'{BASE02}/Iris_02_test_8'),
    #     (f'{BASE02}/Iris_02_test_9'), (f'{BASE02}/Iris_02_test_10'),
    #     (f'{BASE02}/Iris_02_test_11'), (f'{BASE02}/Iris_02_test_12'),
    #     (f'{BASE02}/Iris_02_test_13'), (f'{BASE02}/Iris_02_test_14'),
    #     (f'{BASE02}/Iris_02_test_15'), (f'{BASE02}/Iris_02_test_16'),
    #     (f'{BASE02}/Iris_02_test_17'), (f'{BASE02}/Iris_02_test_18'),
    #     (f'{BASE02}/Iris_02_test_19'), (f'{BASE02}/Iris_02_test_20')
    # ]

    # Test_list = [
    #     (f'{BASE00}/Iris_01_test'),
    #     (f'{BASE00}/Iris_02_test'),
    #     (f'{BASE00}/bc_1_test'),
    #     (f'{BASE00}/bc_2_test'),
    #     (f'{BASE00}/bc_3_test'),
    #     (f'{BASE00}/MNIST_01_test'),
    #     (f'{BASE00}/MNIST_12_test'),
    #     (f'{BASE00}/MNIST_23_test'),
    #     (f'{BASE00}/MNIST_34_test'),
    #     (f'{BASE00}/MNIST_45_test'),
    #     (f'{BASE00}/MNIST_56_test'),
    #     (f'{BASE00}/MNIST_67_test')
    # ]

    Test_list = [
        (f'{BASEBC}/bc_1_test.csv'), (f'{BASEBC}/bc_2_test.csv'),
        (f'{BASEBC}/bc_3_test.csv'), (f'{BASEBC}/bc_4_test.csv'),
        (f'{BASEBC}/bc_5_test.csv'), (f'{BASEBC}/bc_6_test.csv'),
        (f'{BASEBC}/bc_7_test.csv'), (f'{BASEBC}/bc_8_test.csv'),
        (f'{BASEBC}/bc_9_test.csv'), (f'{BASEBC}/bc_10_test.csv'),
        (f'{BASEBC}/bc_11_test.csv'), (f'{BASEBC}/bc_12_test.csv'),
        (f'{BASEBC}/bc_13_test.csv'), (f'{BASEBC}/bc_14_test.csv'),
        (f'{BASEBC}/bc_15_test.csv'), (f'{BASEBC}/bc_16_test.csv'),
        (f'{BASEBC}/bc_17_test.csv'), (f'{BASEBC}/bc_18_test.csv'),
        (f'{BASEBC}/bc_19_test.csv'), (f'{BASEBC}/bc_20_test.csv'),
        (f'{BASEBC}/bc_21_test.csv'), (f'{BASEBC}/bc_22_test.csv'),
        (f'{BASEBC}/bc_23_test.csv'), (f'{BASEBC}/bc_24_test.csv'),
        (f'{BASEBC}/bc_25_test.csv'), (f'{BASEBC}/bc_26_test.csv'),
        (f'{BASEBC}/bc_27_test.csv'), (f'{BASEBC}/bc_28_test.csv'),
        (f'{BASEBC}/bc_29_test.csv'), (f'{BASEBC}/bc_30_test.csv')
    ]
    # holdout_train_csv = f'{BASE03}/Iris_12_2'
    # holdout_test_csv = f'{BASE03}/Iris_12_test_2'
    # holdout_train_csv = f'{BASE00}/Iris_12'
    # holdout_test_csv = f'{BASE00}/Iris_12_test'
    holdout_train_csv = f'{BASEBC}/bc_31.csv'
    holdout_test_csv = f'{BASEBC}/bc_31_test.csv'

    data_list = [
        load_and_preprocess(train_csv, test_csv)
        for train_csv, test_csv in zip(Train_list, Test_list)
    ]
    data_holdout = load_and_preprocess(holdout_train_csv, holdout_test_csv)

    # all_phi_scaled = torch.cat([d.tmc.view(-1) for d in data_list], dim = 0)
    # bias_init = float(all_phi_scaled.mean().item())

    gnn = GINShap(in_channels = 10, hidden_channels = 256, heads = 32)
    loss_history = train(gnn, data_list, epochs = 200, lr = 1e-4, alpha = 0.5)

    plt.figure(figsize = (6, 4))
    plt.plot(loss_history, label = 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Optimisation Curve')
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(PATH_OUT, 'Optimisation_Curve.png')
    plt.savefig(plt_path)
    print('Optimisation Curve Saved.')

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
#df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04/Iris_12_test_2')
#df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_test')
#df_test = pd.read_csv('/tmp/pycharm_project_236/Datasets/TrainnTestSets_V2/Iris_12_test')
df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group05/bc_31_test.csv')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
#df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04/Iris_12_TMC_2')
df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group05/bc_31_TMC.csv')
#df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_TMC')
#df_train = pd.read_csv('/tmp/pycharm_project_236/Datasets/TrainnTestSets_V2/Iris_12_TMC')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','TrueDShap','TMC_Shap'])).to_numpy()
y_train = df_train['target'].to_numpy()
True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_train['TMC_Shap'].to_numpy()
Pred_Shap = preds
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250703/Trial09', 'bc31_Pred_High.png')
#PERF_PATH = os.path.join('/tmp/pycharm_project_236/Outputs/250623/Trial01', 'Iris12Pred_Low.png')
from Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap,
          'Predicted Shapley': Pred_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,105),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)


'Distribution/Magnitude Comparison'
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250703/Trial09'
#GRAPH_DIR = '/tmp/pycharm_project_236/Outputs/250623/Trial01'
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
plt.savefig(out_path, dpi=500)
plt.tight_layout()
plt.show()



'Check Train Sets'
BASE2 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250703/Trial09'
#BASE2 = '/tmp/pycharm_project_236/Outputs/250623/Trial01'
# Train_list = [
#             (f'{BASE2}/Iris_01_1_Pred.csv'), (f'{BASE2}/Iris_01_2_Pred.csv'),
#             (f'{BASE2}/Iris_01_3_Pred.csv'), (f'{BASE2}/Iris_01_4_Pred.csv'),
#             (f'{BASE2}/Iris_01_5_Pred.csv'), (f'{BASE2}/Iris_01_6_Pred.csv'),
#             (f'{BASE2}/Iris_01_7_Pred.csv'), (f'{BASE2}/Iris_01_8_Pred.csv'),
#             (f'{BASE2}/Iris_01_9_Pred.csv'), (f'{BASE2}/Iris_01_10_Pred.csv'),
#             (f'{BASE2}/Iris_01_11_Pred.csv'), (f'{BASE2}/Iris_01_12_Pred.csv'),
#             (f'{BASE2}/Iris_01_13_Pred.csv'), (f'{BASE2}/Iris_01_14_Pred.csv'),
#             (f'{BASE2}/Iris_01_15_Pred.csv'), (f'{BASE2}/Iris_01_16_Pred.csv'),
#             (f'{BASE2}/Iris_01_17_Pred.csv'), (f'{BASE2}/Iris_01_18_Pred.csv'),
#             (f'{BASE2}/Iris_01_19_Pred.csv'), (f'{BASE2}/Iris_01_20_Pred.csv'),
#             (f'{BASE2}/Iris_01_21_Pred.csv'), (f'{BASE2}/Iris_01_22_Pred.csv'),
#             (f'{BASE2}/Iris_01_23_Pred.csv'), (f'{BASE2}/Iris_01_24_Pred.csv'),
#             (f'{BASE2}/Iris_01_25_Pred.csv'), (f'{BASE2}/Iris_01_26_Pred.csv'),
#             (f'{BASE2}/Iris_01_27_Pred.csv'), (f'{BASE2}/Iris_01_28_Pred.csv'),
#             (f'{BASE2}/Iris_01_29_Pred.csv'), (f'{BASE2}/Iris_01_30_Pred.csv'),
#             (f'{BASE2}/Iris_02_1_Pred.csv'), (f'{BASE2}/Iris_02_2_Pred.csv'),
#             (f'{BASE2}/Iris_02_3_Pred.csv'), (f'{BASE2}/Iris_02_4_Pred.csv'),
#             (f'{BASE2}/Iris_02_5_Pred.csv'), (f'{BASE2}/Iris_02_6_Pred.csv'),
#             (f'{BASE2}/Iris_02_7_Pred.csv'), (f'{BASE2}/Iris_02_8_Pred.csv'),
#             (f'{BASE2}/Iris_02_9_Pred.csv'), (f'{BASE2}/Iris_02_10_Pred.csv'),
#             (f'{BASE2}/Iris_02_11_Pred.csv'), (f'{BASE2}/Iris_02_12_Pred.csv'),
#             (f'{BASE2}/Iris_02_13_Pred.csv'), (f'{BASE2}/Iris_02_14_Pred.csv'),
#             (f'{BASE2}/Iris_02_15_Pred.csv'), (f'{BASE2}/Iris_02_16_Pred.csv'),
#             (f'{BASE2}/Iris_02_17_Pred.csv'), (f'{BASE2}/Iris_02_18_Pred.csv'),
#             (f'{BASE2}/Iris_02_19_Pred.csv'), (f'{BASE2}/Iris_02_20_Pred.csv')
# ]
Train_list = [
    (f'{BASE2}/bc_1.csv_Pred.csv'), (f'{BASE2}/bc_2.csv_Pred.csv'),
    (f'{BASE2}/bc_3.csv_Pred.csv'), (f'{BASE2}/bc_4.csv_Pred.csv'),
    (f'{BASE2}/bc_5.csv_Pred.csv'), (f'{BASE2}/bc_6.csv_Pred.csv'),
    (f'{BASE2}/bc_7.csv_Pred.csv'), (f'{BASE2}/bc_8.csv_Pred.csv'),
    (f'{BASE2}/bc_9.csv_Pred.csv'), (f'{BASE2}/bc_10.csv_Pred.csv'),
    (f'{BASE2}/bc_11.csv_Pred.csv'), (f'{BASE2}/bc_12.csv_Pred.csv'),
    (f'{BASE2}/bc_13.csv_Pred.csv'), (f'{BASE2}/bc_14.csv_Pred.csv'),
    (f'{BASE2}/bc_15.csv_Pred.csv'), (f'{BASE2}/bc_16.csv_Pred.csv'),
    (f'{BASE2}/bc_17.csv_Pred.csv'), (f'{BASE2}/bc_18.csv_Pred.csv'),
    (f'{BASE2}/bc_19.csv_Pred.csv'), (f'{BASE2}/bc_20.csv_Pred.csv'),
    (f'{BASE2}/bc_21.csv_Pred.csv'), (f'{BASE2}/bc_22.csv_Pred.csv'),
    (f'{BASE2}/bc_23.csv_Pred.csv'), (f'{BASE2}/bc_24.csv_Pred.csv'),
    (f'{BASE2}/bc_25.csv_Pred.csv'), (f'{BASE2}/bc_26.csv_Pred.csv'),
    (f'{BASE2}/bc_27.csv_Pred.csv'), (f'{BASE2}/bc_28.csv_Pred.csv'),
    (f'{BASE2}/bc_29.csv_Pred.csv'), (f'{BASE2}/bc_30.csv_Pred.csv')

]
# Train_list = [
#             (f'{BASE2}/Iris_01_Pred.csv'),
#             (f'{BASE2}/Iris_02_Pred.csv'),
#             (f'{BASE2}/bc_1_Pred.csv'),
#             (f'{BASE2}/bc_2_Pred.csv'),
#             (f'{BASE2}/bc_3_Pred.csv'),
#             (f'{BASE2}/MNIST_01_Pred.csv'),
#             (f'{BASE2}/MNIST_12_Pred.csv'),
#             (f'{BASE2}/MNIST_23_Pred.csv'),
#             (f'{BASE2}/MNIST_34_Pred.csv'),
#             (f'{BASE2}/MNIST_45_Pred.csv'),
#             (f'{BASE2}/MNIST_56_Pred.csv'),
#             (f'{BASE2}/MNIST_67_Pred.csv')
# ]

for train in Train_list:
    df = pd.read_csv(train)
    true_vals = df['TrueDShap']
    True_Shap = true_vals.to_numpy().reshape(-1, 1).flatten()
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
    order = 'ascending', random_iters = 10,
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