'''
0. Import Packages
'''
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl
from dgl.nn.pytorch.conv import GINConv
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.nn import (Sequential, Linear, LeakyReLU, ReLU, TransformerEncoder,
                      TransformerEncoderLayer, BatchNorm1d, LayerNorm)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd

# from Scripts.Additional_Scripts.Permutation_Shapley import Per_DShap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

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

def SVD_Transform(A, d=40):
    U, S, _ = randomized_svd(A, n_components=d, random_state = 42) #change to full svd/torch svd
    X_svd = U * np.sqrt(S)
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = filp_pe_sign(X_t, method='sum')  # l0 (positive) / sum
    return X_t.numpy()

# def y_SVDTrans(y_raw, d=15):
#     y = np.array(y_raw)
#     sim_matrix = (y[:, None] == y[None, :]).astype(float)
#     U, S, _ = randomized_svd(sim_matrix, n_components = d, random_state = 42)
#     y_svd = U * np.sqrt(S)
#     y_svd = torch.tensor(y_svd, dtype=torch.float)
#     y_svd = filp_pe_sign(y_svd, method='sum')  # l0 (positive) / sum
#     return y_svd.numpy()

'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, phi, total_util, y_raw, c, d = 40):
    # X_raw = np.concatenate([X_raw, y_raw.reshape(-1, 1)], axis = 1)
    D = pairwise_distances(X_raw, metric='euclidean')
    sigma = np.mean(D)
    A = np.exp(-(D ** 2) / (2 * (c * sigma) ** 2))
    X_svd = SVD_Transform(A, d = d)
    #y_svd = y_SVDTrans(y_raw, d = d)
    X_feat = X_svd
    # X_feat = np.concatenate([X_svd, y_svd], axis = 1)
    y = np.asarray(y_raw).ravel()
    A_y = (y[:, None] == y[None, :]).astype(float)

    N = X_raw.shape[0]
    src, dst = np.triu_indices(N, k=1)
    eidx = np.vstack([src, dst])
    edge_index = torch.tensor(np.concatenate([eidx, eidx[::-1]], axis = 1), dtype = torch.long, device=device)
    # w = np.concatenate([A[src, dst], A[src, dst]])
    w_feat = A[src, dst]
    w_label = A_y[src, dst]
    # edge_attr = torch.tensor(w, dtype = torch.float).unsqueeze(1)
    edge_attr_feat = torch.tensor(np.concatenate([w_feat, w_feat]), dtype = torch.float, device=device).unsqueeze(1)
    edge_attr_label = torch.tensor(np.concatenate([w_label, w_label]), dtype = torch.float, device=device).unsqueeze(1)

    data = Data(
        x = torch.tensor(X_feat, dtype = torch.float, device=device),
        edge_index = edge_index,
        edge_attr_feat = edge_attr_feat,
        edge_attr_label = edge_attr_label
    )
    data.X_raw = torch.tensor(X_raw, dtype = torch.float, device=device)
    data.tmc = torch.tensor(phi, dtype = torch.float, device=device)
    data.total_util = torch.tensor(total_util, dtype = torch.float, device=device)
    data.y = torch.tensor(y_raw, dtype = torch.long, device=device)

    return data


'''
3. Data Preprocessing
'''
def load_and_preprocess(train_csv, test_csv, d=40):
    df_train = pd.read_csv(train_csv)
    #X_raw = pd.get_dummies(df_train.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
    X_raw = pd.get_dummies(df_train.drop(columns=['target','PerDShap'])).astype(float).to_numpy()
    # Attached target variable
    y_raw = df_train['target'].to_numpy().astype(int)
    # phi = df_train['TrueDShap'].to_numpy()
    phi = df_train['PerDShap'].to_numpy()
    df_test = pd.read_csv(test_csv)
    X_test = pd.get_dummies(df_test.drop(columns=['target'])).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy()
    # maj = pd.Series(y_test).mode().iloc[0]
    # baseline = (y_test == maj).astype(float)
    # baseline_acc = baseline.mean()
    lr_full = LogisticRegression(max_iter=100, random_state=42)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    data_list = []
    for c in C_List:
        data = build_graph(X_raw, phi, total_util, y_raw, d = d, c = c)
        data_list.append(data)

    return data_list


'''
4. Model Construction
'''
# ''' Version A. Without BatchNorm '''
# class GINShap(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels = 256, out_channels = 1, heads = 8):
#         super().__init__()
#         def mlp(i, o):
#             return Sequential(Linear(i, o), LeakyReLU(), Linear(o, o))
#
#         # 5 Parallel GIN Branches
#         self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
#         self.branches = torch.nn.ModuleList()
#         for _ in C_List:
#             convs = torch.nn.ModuleList()
#             for layer in range(3):
#                 convs.append(
#                     GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
#                             aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
#                             activation = LeakyReLU())
#                 )
#             self.branches.append(convs)
#
#         self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))
#
#         # GT (via TransformerEncoder)
#         d_concat = hidden_channels * len(C_List) # 64 or 128 * 5 = 640
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
#         # ew = data.edge_attr.view(-1) # Extract edge weights for DGL
#
#         # DGL Graph
#         g = dgl.graph((ei[0], ei[1]), num_nodes = x0.size(0))
#         # g.edata['w'] = ew
#
#         # Parallel GIN
#         output_GIN = []
#         # for convs, gdata in zip(self.branches, data_list):
#         #     node_input = gdata.x.to(device)
#         #     for conv in convs:
#         #         node_input = conv(g, node_input , edge_weight = g.edata['w'])
#         #     output_GIN.append(node_input)
#         # output_GIN = torch.cat(output_GIN, dim = 1)
#         for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
#             node_input = gdata.x.to(device)
#             edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)  # (E,)
#             edge_attr_label = gdata.edge_attr_label.view(-1).to(device)  # (E,)
#
#             weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
#             edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label  # (E,)
#
#             g = dgl.graph(
#                 (gdata.edge_index[0].to(device), gdata.edge_index[1].to(device)),
#                 num_nodes=node_input.size(0),
#                 device=device
#             )
#             g.edata['w'] = edge_weight
#
#             for conv in convs:
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

''' Version B. With BatchNorm '''
class GINShap(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels = 256, out_channels = 1, heads = 8):
        super().__init__()
        def mlp(i, o):
            # return Sequential(Linear(i, o), BatchNorm1d(o), LeakyReLU(), Linear(o, o))
            return Sequential(Linear(i, o), LayerNorm(o), LeakyReLU(), Linear(o, o))
        # 5 Parallel GIN Branches
        self.alphas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.betas = torch.nn.Parameter(torch.ones(len(C_List)))
        self.branches = torch.nn.ModuleList()
        for _ in C_List:
            convs = torch.nn.ModuleList()
            for layer in range(5):
                convs.append(
                    GINConv(mlp(in_channels if layer == 0 else hidden_channels, hidden_channels),
                            aggregator_type = 'sum', init_eps = 0.1, learn_eps = True,
                            activation = LeakyReLU())
                )
            self.branches.append(convs)

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))

        # GT (via TransformerEncoder)
        d_concat = hidden_channels * len(C_List) # 64 or 128 * 5 = 640
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
        self.gt = TransformerEncoder(encoder_layer, num_layers = 5)

        # Z
        self.Z = Sequential(Linear(d_concat, hidden_channels), LeakyReLU())

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
        # ew = data.edge_attr.view(-1) # Extract edge weights for DGL

        # DGL Graph
        g = dgl.graph((ei[0], ei[1]), num_nodes = x0.size(0))
        # g.edata['w'] = ew

        # Parallel GIN
        output_GIN = []
        # for convs, gdata in zip(self.branches, data_list):
        #     node_input = gdata.x.to(device)
        #     for conv in convs:
        #         node_input = conv(g, node_input , edge_weight = g.edata['w'])
        #     output_GIN.append(node_input)
        # output_GIN = torch.cat(output_GIN, dim = 1)
        for idx, (convs, gdata) in enumerate(zip(self.branches, data_list)):
            node_input = gdata.x.to(device)
            edge_attr_feat = gdata.edge_attr_feat.view(-1).to(device)  # (E,)
            edge_attr_label = gdata.edge_attr_label.view(-1).to(device)  # (E,)

            weights = F.softmax(torch.stack([self.alphas[idx], self.betas[idx]]), dim=0)
            edge_weight = weights[0] * edge_attr_feat + weights[1] * edge_attr_label  # (E,)

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
        output_skip = output_skip.unsqueeze(0) # transform to: (1, N, hidden_channels*5)
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

def train(model, dataset_graphs, epochs=500, lr=1e-5):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # adamw
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs) # cosine
    #scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
    # lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for data_list in dataset_graphs:
            phi_hat = model(data_list).view(-1)
            phi_true = data_list[0].tmc.to(device)
            # # Add l1 normalisation to ensure equal contributions of datasets with different number of points
            # phi_true_scaled = phi_true/phi_true.sum()
            # phi_hat_scaled = phi_hat / phi_hat.sum()
            # loss_MSE = F.mse_loss(phi_hat_scaled, phi_true_scaled, reduction = 'sum')

            # loss_MSE = F.mse_loss(phi_hat, phi_true, reduction='mean') * (len(phi_true) ** 2)
            # loss_MSE = F.mse_loss(F.softmax(phi_hat, dim=0), F.softmax(phi_true, dim=0))
            # loss_MSE = F.mse_loss(phi_hat_scaled, phi_true_scaled, reduction='sum') / len(phi_true)
            # loss_Rank = BPR_loss(phi_hat_scaled, phi_true_scaled)
            # loss = (1-alpha) * loss_MSE + alpha * loss_Rank

            # loss_MSE = F.l1_loss(phi_hat, phi_true, reduction='sum')
            # loss_MSE = F.mse_loss(phi_hat, phi_true, reduction='mean') * (len(phi_true) ** 2)
            loss_MSE = F.mse_loss(phi_hat, phi_true, reduction='mean')

            loss = loss_MSE
            losses.append(loss)
        L = torch.stack(losses).mean()
        loss_history.append(L.item())

        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        # scheduler.step()

        if ep % 5 == 0:
            # current_lr = scheduler.get_last_lr()[0]
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {ep:3d} | total loss {L:.16f}| LR: {current_lr:.2e}')

    return loss_history


'''
6. Prediction
'''
def predict(model, data_list):
    model.eval()
    # model.train()
    d = data_list[0].to(device)
    with torch.no_grad():
        return model(data_list).squeeze().cpu().numpy()



######################################################################################################
######################################################################################################
#########################################  Experiments ###############################################

'''
EXP 1
'''

'A. Implementation'
# if __name__ == '__main__':
#     BASE00 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2'
#     BASE01 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group01'
#     BASE02 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group03'
#     BASE03 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04'
#     BASEBC = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group05'
#     #BASE = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V2'
#     PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250703/Trial09'
#     #PATH_OUT = '/tmp/pycharm_project_236/Outputs/250623/Trial01'
#     # Train_list = [
#     #     (f'{BASE01}/Iris_01_1'),(f'{BASE01}/Iris_01_2'),
#     #     (f'{BASE01}/Iris_01_3'),(f'{BASE01}/Iris_01_4'),
#     #     (f'{BASE01}/Iris_01_5'),(f'{BASE01}/Iris_01_6'),
#     #     (f'{BASE01}/Iris_01_7'),(f'{BASE01}/Iris_01_8'),
#     #     (f'{BASE01}/Iris_01_9'),(f'{BASE01}/Iris_01_10'),
#     #     (f'{BASE01}/Iris_01_11'),(f'{BASE01}/Iris_01_12'),
#     #     (f'{BASE01}/Iris_01_13'),(f'{BASE01}/Iris_01_14'),
#     #     (f'{BASE01}/Iris_01_15'),(f'{BASE01}/Iris_01_16'),
#     #     (f'{BASE01}/Iris_01_17'), (f'{BASE01}/Iris_01_18'),
#     #     (f'{BASE01}/Iris_01_19'), (f'{BASE01}/Iris_01_20'),
#     #     (f'{BASE01}/Iris_01_21'), (f'{BASE01}/Iris_01_22'),
#     #     (f'{BASE01}/Iris_01_23'), (f'{BASE01}/Iris_01_24'),
#     #     (f'{BASE01}/Iris_01_25'), (f'{BASE01}/Iris_01_26'),
#     #     (f'{BASE01}/Iris_01_27'), (f'{BASE01}/Iris_01_28'),
#     #     (f'{BASE01}/Iris_01_29'), (f'{BASE01}/Iris_01_30'),
#     #     (f'{BASE02}/Iris_02_1'), (f'{BASE02}/Iris_02_2'),
#     #     (f'{BASE02}/Iris_02_3'), (f'{BASE02}/Iris_02_4'),
#     #     (f'{BASE02}/Iris_02_5'), (f'{BASE02}/Iris_02_6'),
#     #     (f'{BASE02}/Iris_02_7'), (f'{BASE02}/Iris_02_8'),
#     #     (f'{BASE02}/Iris_02_9'), (f'{BASE02}/Iris_02_10'),
#     #     (f'{BASE02}/Iris_02_11'), (f'{BASE02}/Iris_02_12'),
#     #     (f'{BASE02}/Iris_02_13'), (f'{BASE02}/Iris_02_14'),
#     #     (f'{BASE02}/Iris_02_15'), (f'{BASE02}/Iris_02_16'),
#     #     (f'{BASE02}/Iris_02_17'), (f'{BASE02}/Iris_02_18'),
#     #     (f'{BASE02}/Iris_02_19'), (f'{BASE02}/Iris_02_20')
#     # ]
#     # Train_list = [
#     #     (f'{BASE00}/Iris_01'),
#     #     (f'{BASE00}/Iris_02'),
#     #     (f'{BASE00}/bc_1'),
#     #     (f'{BASE00}/bc_2'),
#     #     (f'{BASE00}/bc_3'),
#     #     (f'{BASE00}/MNIST_01'),
#     #     (f'{BASE00}/MNIST_12'),
#     #     (f'{BASE00}/MNIST_23'),
#     #     (f'{BASE00}/MNIST_34'),
#     #     (f'{BASE00}/MNIST_45'),
#     #     (f'{BASE00}/MNIST_56'),
#     #     (f'{BASE00}/MNIST_67')
#     # ]
#     Train_list = [
#         (f'{BASEBC}/bc_1.csv'), (f'{BASEBC}/bc_2.csv'),
#         (f'{BASEBC}/bc_3.csv'), (f'{BASEBC}/bc_4.csv'),
#         (f'{BASEBC}/bc_5.csv'), (f'{BASEBC}/bc_6.csv'),
#         (f'{BASEBC}/bc_7.csv'), (f'{BASEBC}/bc_8.csv'),
#         (f'{BASEBC}/bc_9.csv'), (f'{BASEBC}/bc_10.csv'),
#         (f'{BASEBC}/bc_11.csv'), (f'{BASEBC}/bc_12.csv'),
#         (f'{BASEBC}/bc_13.csv'), (f'{BASEBC}/bc_14.csv'),
#         (f'{BASEBC}/bc_15.csv'), (f'{BASEBC}/bc_16.csv'),
#         (f'{BASEBC}/bc_17.csv'), (f'{BASEBC}/bc_18.csv'),
#         (f'{BASEBC}/bc_19.csv'), (f'{BASEBC}/bc_20.csv'),
#         (f'{BASEBC}/bc_21.csv'), (f'{BASEBC}/bc_22.csv'),
#         (f'{BASEBC}/bc_23.csv'), (f'{BASEBC}/bc_24.csv'),
#         (f'{BASEBC}/bc_25.csv'), (f'{BASEBC}/bc_26.csv'),
#         (f'{BASEBC}/bc_27.csv'), (f'{BASEBC}/bc_28.csv'),
#         (f'{BASEBC}/bc_29.csv'), (f'{BASEBC}/bc_30.csv')
#     ]
#
#     # Test_list = [
#     #     (f'{BASE01}/Iris_01_test_1'),(f'{BASE01}/Iris_01_test_2'),
#     #     (f'{BASE01}/Iris_01_test_3'),(f'{BASE01}/Iris_01_test_4'),
#     #     (f'{BASE01}/Iris_01_test_5'),(f'{BASE01}/Iris_01_test_6'),
#     #     (f'{BASE01}/Iris_01_test_7'),(f'{BASE01}/Iris_01_test_8'),
#     #     (f'{BASE01}/Iris_01_test_9'),(f'{BASE01}/Iris_01_test_10'),
#     #     (f'{BASE01}/Iris_01_test_11'),(f'{BASE01}/Iris_01_test_12'),
#     #     (f'{BASE01}/Iris_01_test_13'),(f'{BASE01}/Iris_01_test_14'),
#     #     (f'{BASE01}/Iris_01_test_15'),(f'{BASE01}/Iris_01_test_16'),
#     #     (f'{BASE01}/Iris_01_test_17'), (f'{BASE01}/Iris_01_test_18'),
#     #     (f'{BASE01}/Iris_01_test_19'), (f'{BASE01}/Iris_01_test_20'),
#     #     (f'{BASE01}/Iris_01_test_21'), (f'{BASE01}/Iris_01_test_22'),
#     #     (f'{BASE01}/Iris_01_test_23'), (f'{BASE01}/Iris_01_test_24'),
#     #     (f'{BASE01}/Iris_01_test_25'), (f'{BASE01}/Iris_01_test_26'),
#     #     (f'{BASE01}/Iris_01_test_27'), (f'{BASE01}/Iris_01_test_28'),
#     #     (f'{BASE01}/Iris_01_test_29'), (f'{BASE01}/Iris_01_test_30'),
#     #     (f'{BASE02}/Iris_02_test_1'), (f'{BASE02}/Iris_02_test_2'),
#     #     (f'{BASE02}/Iris_02_test_3'), (f'{BASE02}/Iris_02_test_4'),
#     #     (f'{BASE02}/Iris_02_test_5'), (f'{BASE02}/Iris_02_test_6'),
#     #     (f'{BASE02}/Iris_02_test_7'), (f'{BASE02}/Iris_02_test_8'),
#     #     (f'{BASE02}/Iris_02_test_9'), (f'{BASE02}/Iris_02_test_10'),
#     #     (f'{BASE02}/Iris_02_test_11'), (f'{BASE02}/Iris_02_test_12'),
#     #     (f'{BASE02}/Iris_02_test_13'), (f'{BASE02}/Iris_02_test_14'),
#     #     (f'{BASE02}/Iris_02_test_15'), (f'{BASE02}/Iris_02_test_16'),
#     #     (f'{BASE02}/Iris_02_test_17'), (f'{BASE02}/Iris_02_test_18'),
#     #     (f'{BASE02}/Iris_02_test_19'), (f'{BASE02}/Iris_02_test_20')
#     # ]
#
#     # Test_list = [
#     #     (f'{BASE00}/Iris_01_test'),
#     #     (f'{BASE00}/Iris_02_test'),
#     #     (f'{BASE00}/bc_1_test'),
#     #     (f'{BASE00}/bc_2_test'),
#     #     (f'{BASE00}/bc_3_test'),
#     #     (f'{BASE00}/MNIST_01_test'),
#     #     (f'{BASE00}/MNIST_12_test'),
#     #     (f'{BASE00}/MNIST_23_test'),
#     #     (f'{BASE00}/MNIST_34_test'),
#     #     (f'{BASE00}/MNIST_45_test'),
#     #     (f'{BASE00}/MNIST_56_test'),
#     #     (f'{BASE00}/MNIST_67_test')
#     # ]
#
#     Test_list = [
#         (f'{BASEBC}/bc_1_test.csv'), (f'{BASEBC}/bc_2_test.csv'),
#         (f'{BASEBC}/bc_3_test.csv'), (f'{BASEBC}/bc_4_test.csv'),
#         (f'{BASEBC}/bc_5_test.csv'), (f'{BASEBC}/bc_6_test.csv'),
#         (f'{BASEBC}/bc_7_test.csv'), (f'{BASEBC}/bc_8_test.csv'),
#         (f'{BASEBC}/bc_9_test.csv'), (f'{BASEBC}/bc_10_test.csv'),
#         (f'{BASEBC}/bc_11_test.csv'), (f'{BASEBC}/bc_12_test.csv'),
#         (f'{BASEBC}/bc_13_test.csv'), (f'{BASEBC}/bc_14_test.csv'),
#         (f'{BASEBC}/bc_15_test.csv'), (f'{BASEBC}/bc_16_test.csv'),
#         (f'{BASEBC}/bc_17_test.csv'), (f'{BASEBC}/bc_18_test.csv'),
#         (f'{BASEBC}/bc_19_test.csv'), (f'{BASEBC}/bc_20_test.csv'),
#         (f'{BASEBC}/bc_21_test.csv'), (f'{BASEBC}/bc_22_test.csv'),
#         (f'{BASEBC}/bc_23_test.csv'), (f'{BASEBC}/bc_24_test.csv'),
#         (f'{BASEBC}/bc_25_test.csv'), (f'{BASEBC}/bc_26_test.csv'),
#         (f'{BASEBC}/bc_27_test.csv'), (f'{BASEBC}/bc_28_test.csv'),
#         (f'{BASEBC}/bc_29_test.csv'), (f'{BASEBC}/bc_30_test.csv')
#     ]
#     # holdout_train_csv = f'{BASE03}/Iris_12_2'
#     # holdout_test_csv = f'{BASE03}/Iris_12_test_2'
#     # holdout_train_csv = f'{BASE00}/Iris_12'
#     # holdout_test_csv = f'{BASE00}/Iris_12_test'
#     holdout_train_csv = f'{BASEBC}/bc_31.csv'
#     holdout_test_csv = f'{BASEBC}/bc_31_test.csv'
#
#     data_list = [
#         load_and_preprocess(train_csv, test_csv)
#         for train_csv, test_csv in zip(Train_list, Test_list)
#     ]
#     data_holdout = load_and_preprocess(holdout_train_csv, holdout_test_csv)
#
#     # all_phi_scaled = torch.cat([d.tmc.view(-1) for d in data_list], dim = 0)
#     # bias_init = float(all_phi_scaled.mean().item())
#
#     gnn = GINShap(in_channels = 10, hidden_channels = 256, heads = 32)
#     loss_history = train(gnn, data_list, epochs = 200, lr = 1e-4, alpha = 0.5)
#
#     plt.figure(figsize = (6, 4))
#     plt.plot(loss_history, label = 'Training Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Optimisation Curve')
#     plt.legend()
#     plt.grid(True)
#     plt_path = os.path.join(PATH_OUT, 'Optimisation_Curve.png')
#     plt.savefig(plt_path)
#     print('Optimisation Curve Saved.')
#
#     preds = predict(gnn, data_holdout)
#     print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))
#
#     df_holdout = pd.read_csv(holdout_train_csv)
#     df_holdout['Pred_Shap'] = preds
#     out_csv = f'{PATH_OUT}/Iris_12_Pred.csv'
#     df_holdout.to_csv(out_csv, index=False)
#     print('saved ➜', out_csv)
#
#     # Predict and save for each training dataset
#     for train_csv, data in zip(Train_list, data_list):
#         preds_train = predict(gnn, data)
#
#         df_train = pd.read_csv(train_csv)
#         df_train['Pred_Shap'] = preds_train
#
#         base_name = os.path.basename(train_csv)
#         out_csv_train = os.path.join(PATH_OUT, f'{base_name}_Pred.csv')
#         df_train.to_csv(out_csv_train, index=False)
#         print('saved ➜', out_csv_train)

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

if __name__ == '__main__':
    BASE00 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Train'
    BASE11 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test'
    PATH_OUT = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250910/Trial09/'
    # BASE00 = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V6/10n15p20/Train'
    # BASE11 = '/tmp/pycharm_project_236/Datasets/TrainnTestSets_V6/10n15p20/Test'
    # PATH_OUT = '/tmp/pycharm_project_236/Outputs/250821/Trial01/'

    base_dirs = [BASE00]
    all_pairs = []
    for bd in base_dirs:
        all_pairs += find_train_test_pairs(bd)

    Train_list, Test_list = zip(*all_pairs)
    holdout_train_csv = os.path.join(BASE11, 'MNIST_17_1')
    holdout_test_csv  = os.path.join(BASE11, 'MNIST_17_test_1')

    # load everything
    data_list    = [load_and_preprocess(train, test)
                    for train, test in zip(Train_list, Test_list)]
    data_holdout = load_and_preprocess(holdout_train_csv, holdout_test_csv)

    # build & train
    gnn = GINShap(in_channels=40, hidden_channels=256, heads=8)
    loss_history = train(gnn, data_list, epochs=3000, lr=1e-5)

    # plot optimisation curve
    plt.figure(figsize=(6,4))
    plt.plot(np.log(loss_history), label='Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('log(Loss)')
    plt.title('Optimisation Curve(Log)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PATH_OUT, 'Optimisation_Curve.png'))

    # holdout predictions
    preds = predict(gnn, data_holdout)

    print('prediction percentiles:', np.percentile(preds,[0,5,50,95,100]))

    # save holdout
    df_holdout = pd.read_csv(holdout_train_csv)
    df_holdout['Pred_Shap'] = preds
    df_holdout.to_csv(os.path.join(PATH_OUT, 'Iris_02_21_Pred.csv'), index=False)

    # save train preds
    for train_csv, data in zip(Train_list, data_list):
        df_train = pd.read_csv(train_csv)
        df_train['Pred_Shap'] = predict(gnn, data)
        out = os.path.join(PATH_OUT, os.path.basename(train_csv).replace('.csv','') + '_Pred.csv')
        df_train.to_csv(out, index=False)
        print('saved ➜', out)


'Removal Plots'
#df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04/Iris_12_test_2')
#df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_test')
# df_test = pd.read_csv('/tmp/pycharm_project_236/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_test_1')
df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_test_1')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
#df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V3/Group04/Iris_12_TMC_2')
df_tmc = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_TMC_1')
df_train = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_1')
#df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_TMC')
# df_train = pd.read_csv('/tmp/pycharm_project_236/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_1')
# df_tmc = pd.read_csv('/tmp/pycharm_project_236/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_TMC_1')
# X_train = pd.get_dummies(df_train.drop(
#                 columns=['target','TrueDShap','TMC_Shap'])).to_numpy()
X_train = pd.get_dummies(df_train.drop(
                columns=['target','PerDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
# True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
Per_DShap = df_train['PerDShap'].to_numpy()
Pred_Shap = preds
# df_holdout = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250907/Trial01/Iris_02_1_Pred.csv')
# preds = df_holdout['Pred_Shap']
# PERF_PATH = os.path.join('/tmp/pycharm_project_236/Outputs/250826/Trial01/', 'MNIST17_1_High.png')
#PERF_PATH = os.path.join('/tmp/pycharm_project_236/Outputs/250715/Trial04', 'MNIST17_1_Low.png')
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250910/Trial09/', 'MNIST_17_1_High.png')
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={ 'Permuted Data Shapley': Per_DShap,
        # 'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap,
          'Predicted Shapley': Pred_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,105),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)


'Distribution/Magnitude Comparison'
GRAPH_DIR = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250910/Trial09'
# GRAPH_DIR = '/tmp/pycharm_project_236/Outputs/250814/Trial02/'
# true_vals = np.asarray(True_Shap)
tmc_vals  = np.asarray(TMC_Shap)
per_vals = np.asarray(Per_DShap)
pred_vals = np.asarray(Pred_Shap)
# mean_true = true_vals.mean()
mean_tmc  = tmc_vals.mean()
mean_per = per_vals.mean()
mean_pred = pred_vals.mean()
# N = len(true_vals)
N = len(Per_DShap)
indices = np.arange(N)
width = 0.25
plt.figure(figsize=(10,6))
# plt.bar(indices - width, true_vals,  width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
plt.bar(indices, tmc_vals, width, label='TMC Shapley', color='tab:orange',edgecolor='k', alpha=0.7)
plt.bar(indices, per_vals, width, label='Permuted Shapley', color='tab:purple',edgecolor='k', alpha=0.7)
plt.bar(indices + width, pred_vals, width, label='Predicted Shapley',  color='tab:green', edgecolor='k', alpha=0.7)
# plt.plot(indices, [mean_true]*N, linestyle='--', linewidth=2, label=f'Average True Shapley Values ({mean_true:.3f})',   color='tab:blue')
plt.plot(indices, [mean_tmc]*N, linestyle='--', linewidth=2, label=f'Average TMC Shapley Values ({mean_tmc:.3f})',     color='tab:orange')
plt.plot(indices, [mean_per]*N, linestyle='--', linewidth=2, label=f'Average Permuted Shapley Values ({mean_per:.3f})', color='tab:purple')
plt.plot(indices, [mean_pred]*N, linestyle='--', linewidth=2, label=f'Average Predicted Shapley Values ({mean_pred:.3f})',color='tab:green')
plt.xlabel('Sample Index', fontsize=12, weight='bold')
plt.ylabel('Shapley Value', fontsize=12, weight='bold')
# plt.title('Comparison of True, TMC & Predicted Shapley Values', fontsize=14, weight='bold')
plt.title('Comparison of TMC & Permuted & Predicted Shapley Values', fontsize=14, weight='bold')
plt.xticks(indices)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, 'Shapley_compare.png')
plt.savefig(out_path, dpi=500)
plt.show()

'Linear Plot'
# true_vals = np.asarray(True_Shap)
pred_vals = np.asarray(Pred_Shap)
tmc_vals  = np.asarray(TMC_Shap)
per_vals = np.asarray(Per_DShap)

plt.figure(figsize=(8,8))
# plt.scatter(true_vals, pred_vals,
#             s=200, color='tab:blue', alpha=0.6, edgecolors='k',
#             linewidth=0.5, label='Predicted Shapley')
plt.scatter(per_vals, tmc_vals,
            s=200, color='tab:orange', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='TMC Shapley')
plt.scatter(per_vals, pred_vals,
            s=200, color='tab:purple', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='Predicted vs Permuted Shapley')
# all_min = min(true_vals.min(), pred_vals.min(), tmc_vals.min())
# all_max = max(true_vals.max(), pred_vals.max(), tmc_vals.max())
all_min = min(per_vals.min(),tmc_vals.mean(), pred_vals.min())
all_max = max(per_vals.max(),tmc_vals.mean(), pred_vals.max())
plt.plot([all_min - 0.01, all_max + 0.01], [all_min-0.01, all_max+0.01],'--k', linewidth=1.5, label='y = x')

# plt.xlabel('True TMC Shapley Values', fontsize=10, weight='bold')
plt.xlabel('Permuted Shapley Value', fontsize=12, weight='bold')
plt.ylabel('Predicted Shapley Values', fontsize=10, weight='bold')
# plt.title('True vs Predicted & TMC Shapley Values', fontsize=12, weight='bold')
plt.title('Permuted vs TMC & Predicted Shapley Values', fontsize=14, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

out_path = os.path.join(GRAPH_DIR, 'DShap_LinearCompare2.png')
plt.savefig(out_path, dpi=500)
plt.tight_layout()
plt.show()



'Check Train Sets'
BASE2 = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250910/Trial09'
# BASE2 = '/tmp/pycharm_project_236/Outputs/250815/Trial02'
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
#             (f'{BASE2}/Iris_01_23_Pred.csv), (f'{BASE2}/Iris_01_24_Pred.csv'),
#             (f'{BASE2}/Iris_01_25_Pred.csv), (f'{BASE2}/Iris_01_26_Pred.csv'),
#             (f'{BASE2}/Iris_01_27_Pred.csv), (f'{BASE2}/Iris_01_28_Pred.csv'),
#             (f'{BASE2}/Iris_01_29_Pred.csv), (f'{BASE2}/Iris_01_30_Pred.csv'),
#             (f'{BASE2}/Iris_02_1_Pred.csv), (f'{BASE2}/Iris_02_2_Pred.csv'),
#             (f'{BASE2}/Iris_02_3_Pred.csv), (f'{BASE2}/Iris_02_4_Pred.csv'),
#             (f'{BASE2}/Iris_02_5_Pred.csv), (f'{BASE2}/Iris_02_6_Pred.csv'),
#             (f'{BASE2}/Iris_02_7_Pred.csv), (f'{BASE2}/Iris_02_8_Pred.csv'),
#             (f'{BASE2}/Iris_02_9_Pred.csv), (f'{BASE2}/Iris_02_10_Pred.csv'),
#             (f'{BASE2}/Iris_02_11_Pred.csv), (f'{BASE2}/Iris_02_12_Pred.csv'),
#             (f'{BASE2}/Iris_02_13_Pred.csv), (f'{BASE2}/Iris_02_14_Pred.csv'),
#             (f'{BASE2}/Iris_02_15_Pred.csv), (f'{BASE2}/Iris_02_16_Pred.csv'),
#             (f'{BASE2}/Iris_02_17_Pred.csv), (f'{BASE2}/Iris_02_18_Pred.csv'),
#             (f'{BASE2}/Iris_02_19_Pred.csv), (f'{BASE2}/Iris_02_20_Pred.csv')
# ]
# Train_list = [
#     (f'{BASE2}/Iris_01_1_Pred.csv'), (f'{BASE2}/Iris_12_1_Pred.csv'),
#     (f'{BASE2}/Iris_01_3_Pred.csv'), (f'{BASE2}/Iris_12_2_Pred.csv'),
#     (f'{BASE2}/Iris_01_5_Pred.csv'), (f'{BASE2}/Iris_12_5_Pred.csv'),
#     (f'{BASE2}/Iris_01_7_Pred.csv'), (f'{BASE2}/Iris_12_8_Pred.csv')

    # (f'{BASE2}/bc_1_Pred.csv'), (f'{BASE2}/bc_2_Pred.csv'),
    # (f'{BASE2}/MNIST_12_1_Pred.csv'), (f'{BASE2}/MNIST_45_1_Pred.csv'),
# ]
Train_list = [
    (f'{BASE2}/bc_1_Pred.csv'), (f'{BASE2}/bc_2_Pred.csv'),
    (f'{BASE2}/Iris_01_1_Pred.csv'), (f'{BASE2}/MNIST_23_1_Pred.csv'),
    (f'{BASE2}/Iris_12_1_Pred.csv'), (f'{BASE2}/MNIST_01_1_Pred.csv'),
    (f'{BASE2}/Iris_02_1_Pred.csv'), (f'{BASE2}/MNIST_12_1_Pred.csv')
]
# Train_list = [
#     (f'{BASE2}/bc_1.csv_Pred.csv'), (f'{BASE2}/bc_2.csv_Pred.csv'),
#     (f'{BASE2}/bc_3.csv_Pred.csv'), (f'{BASE2}/bc_4.csv_Pred.csv'),
#     (f'{BASE2}/bc_5.csv_Pred.csv'), (f'{BASE2}/bc_6.csv_Pred.csv'),
#     (f'{BASE2}/bc_7.csv_Pred.csv'), (f'{BASE2}/bc_8.csv_Pred.csv'),
#     (f'{BASE2}/bc_9.csv_Pred.csv'), (f'{BASE2}/bc_10.csv_Pred.csv'),
#     (f'{BASE2}/bc_11.csv_Pred.csv'), (f'{BASE2}/bc_12.csv_Pred.csv'),
#     (f'{BASE2}/bc_13.csv_Pred.csv'), (f'{BASE2}/bc_14.csv_Pred.csv'),
#     (f'{BASE2}/bc_15.csv_Pred.csv'), (f'{BASE2}/bc_16.csv_Pred.csv'),
#     (f'{BASE2}/bc_17.csv_Pred.csv'), (f'{BASE2}/bc_18.csv_Pred.csv'),
#     (f'{BASE2}/bc_19.csv_Pred.csv'), (f'{BASE2}/bc_20.csv_Pred.csv'),
#     (f'{BASE2}/bc_21.csv_Pred.csv'), (f'{BASE2}/bc_22.csv_Pred.csv'),
#     (f'{BASE2}/bc_23.csv_Pred.csv'), (f'{BASE2}/bc_24.csv_Pred.csv'),
#     (f'{BASE2}/bc_25.csv_Pred.csv'), (f'{BASE2}/bc_26.csv_Pred.csv'),
#     (f'{BASE2}/bc_27.csv_Pred.csv'), (f'{BASE2}/bc_28.csv_Pred.csv'),
#     (f'{BASE2}/bc_29.csv_Pred.csv'), (f'{BASE2}/bc_30.csv_Pred.csv')
#
# ]
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
    # true_vals = df['TrueDShap']
    true_vals = df['PerDShap']
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
