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

from dgl.nn.pytorch.conv import GINConv
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR, LambdaLR
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

def SVD_Transform(A, d=10):
    # U, S, _ = randomized_svd(A, n_components=d, random_state = 23) #change to full svd/torch svd
    # X_svd = U * np.sqrt(S)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    X_svd = U[:, :d] * np.sqrt(S[:d])
    X_t = torch.tensor(X_svd, dtype=torch.float)
    X_t = flip_pe_sign(X_t, method='sum')  # l0 (positive) / sum
    return X_t.numpy()



'''
2. Graph Construction
'''
C_List = [0.25, 0.5, 1, 2, 4]

def build_graph(X_raw, y_raw, phi, total_util, c, d = 10):
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

    return data


'''
3. Data Preprocessing
'''
def load_and_preprocess(train_csv, test_csv, d=10):
    df_train = pd.read_csv(train_csv)
    X_raw = df_train.drop(columns=['target','PerDShap']).astype(float).to_numpy()
    y_raw = df_train['target'].to_numpy().astype(int)
    phi = df_train['PerDShap'].to_numpy()
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target']).astype(float).to_numpy()
    y_test = df_test['target'].to_numpy()

    lr_full = LogisticRegression(max_iter=100, random_state=23)
    lr_full.fit(X_raw, y_raw)
    model_acc = lr_full.score(X_test, y_test)
    total_util = model_acc

    data_list = [] # length = 5
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
            self.branches.append(convs)  # length = 5

        self.skip_lin = Linear(in_channels, hidden_channels * len(C_List))

        # GT (via TransformerEncoder)
        d_concat = hidden_channels * len(C_List)
        encoder_layer = TransformerEncoderLayer(d_model=d_concat, nhead=heads, dropout=0.0, batch_first = True)
        self.gt = TransformerEncoder(encoder_layer, num_layers = 3)

        # Z
        self.Z = Sequential(Linear(d_concat, hidden_channels),ReLU())

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
def train(model, dataset_graphs, epochs=500, lr=1e-5):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # adamw
    loss_history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for data_list in dataset_graphs:
            phi_hat = model(data_list).view(-1)
            phi_true = data_list[0].phi.to(device)
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
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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

def _mk_paths(base_test_dir: str, base_name: str):
    parts = base_name.split("_")
    if len(parts) == 3:
        # MNIST-style: MNIST_{ab}_{i}
        prefix, ab, i = parts[0], parts[1], parts[2]
        train_like_csv = os.path.join(base_test_dir, base_name)
        test_csv       = os.path.join(base_test_dir, f"{prefix}_{ab}_test_{i}")
        tmc_csv        = os.path.join(base_test_dir, f"{prefix}_{ab}_TMC_{i}")
        return dict(test=test_csv, tmc=tmc_csv, train_like=train_like_csv)

    elif len(parts) == 2:
        # Two-part style: <prefix>_{i} (e.g., bc_1)
        prefix, i = parts[0], parts[1]
        train_like_csv = os.path.join(base_test_dir, base_name)
        test_csv       = os.path.join(base_test_dir, f"{prefix}_test_{i}")
        tmc_csv        = os.path.join(base_test_dir, f"{prefix}_TMC_{i}")
        return dict(test=test_csv, tmc=tmc_csv, train_like=train_like_csv)

    else:
        raise ValueError(f"Unexpected base_name format: {base_name}")

def diagnose_train_sets(gnn, base_train_dir: str, out_dir: str, graph_dir: str = None):
    _ensure_dir(out_dir)
    graph_dir = graph_dir or out_dir
    _ensure_dir(graph_dir)

    pairs = find_train_test_pairs(base_train_dir)
    for train_csv, test_csv in pairs:
        dlist = load_and_preprocess(train_csv, test_csv)
        preds = predict(gnn, dlist)

        df_train = pd.read_csv(train_csv)
        df_train['Pred_Shap'] = preds

        base_name = os.path.basename(train_csv).replace('.csv', '')
        out_csv   = os.path.join(out_dir, f'{base_name}_Pred.csv')
        df_train.to_csv(out_csv, index=False)
        print('saved ➜', out_csv)

        true_vals = df_train['PerDShap'].to_numpy()
        pred_vals = df_train['Pred_Shap'].to_numpy()
        mean_true, mean_pred = true_vals.mean(), pred_vals.mean()
        N = len(true_vals); idx = np.arange(N); width = 0.25

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # bar
        ax1.bar(idx - width, true_vals, width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
        ax1.bar(idx + width, pred_vals, width, label='Predicted Shapley', color='tab:green', edgecolor='k', alpha=0.7)
        ax1.plot(idx, [mean_true]*N, '--', linewidth=2, color='tab:blue',  label=f'Mean True: {mean_true:.3f}')
        ax1.plot(idx, [mean_pred]*N, '--', linewidth=2, color='tab:green', label=f'Mean Pred: {mean_pred:.3f}')
        ax1.set_xlabel('Sample Index', fontsize=12, weight='bold')
        ax1.set_ylabel('Shapley Value', fontsize=12, weight='bold')
        ax1.set_title('Normalized True vs Predicted Shapley (Bar)', fontsize=13, weight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.4); ax1.legend(fontsize=10)

        # linear
        lo, hi = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
        ax2.scatter(true_vals, pred_vals, s=80, color='tab:blue', alpha=0.6, edgecolors='k', linewidth=0.5,
                    label='Predicted vs True')
        ax2.plot([lo, hi], [lo, hi], '--k', linewidth=1.5, label='y = x')
        ax2.set_xlabel('True Shapley Values (normalized)', fontsize=12, weight='bold')
        ax2.set_ylabel('Predicted Shapley Values', fontsize=12, weight='bold')
        ax2.set_title('True vs Predicted (Scatter)', fontsize=13, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4); ax2.legend(fontsize=10)

        fig.suptitle(f'Shapley Comparison: {os.path.basename(train_csv)}', fontsize=15, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(graph_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(out_png, dpi=300); plt.close()
        print('saved ➜', out_png)

def diagnose_test_sets(gnn, base_test_dir: str, holdout_bases, out_dir: str,
                       a=0.05, b=0.40, ylim=(40,105), random_iters=10):
    _ensure_dir(out_dir)

    for base_name in holdout_bases:
        paths = _mk_paths(base_test_dir, base_name)

        dlist = load_and_preprocess(paths['train_like'], paths['test'])
        preds = predict(gnn, dlist)

        df_tr_like = pd.read_csv(paths['train_like'])
        df_tr_like['Pred_Shap'] = preds
        out_pred_csv = os.path.join(out_dir, f'{base_name}_Pred.csv')
        df_tr_like.to_csv(out_pred_csv, index=False)
        print('saved ➜', out_pred_csv)

        df_test = pd.read_csv(paths['test'])

        X_train = df_tr_like.drop(columns=['target','PerDShap','Pred_Shap']).astype(float)
        X_test  = df_test.drop(columns=['target']).astype(float)
        X_test  = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        y_train = df_tr_like['target'].to_numpy()
        y_test  = df_test['target'].to_numpy()

        df_tmc   = pd.read_csv(paths['tmc'])
        TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
        Per_DShap = df_tr_like['PerDShap'].to_numpy()
        Pred_Shap = preds

        # Removal Plots
        desc_png = os.path.join(out_dir, f'{base_name}_Removal_Desc.png')
        _ = PerformancePlot_Removal(
            vals={'Permuted Data Shapley': Per_DShap,
                  'TMC Data Shapley': TMC_Shap,
                  'Predicted Shapley': Pred_Shap},
            X_train=X_train.to_numpy(), y_train=y_train,
            X_test=X_test.to_numpy(),   y_test=y_test,
            a=a, b=b, ylim=ylim,
            order='descending', random_iters=random_iters,
            save_path=desc_png
        )
        print('saved ➜', desc_png)

        asc_png = os.path.join(out_dir, f'{base_name}_Removal_Asc.png')
        _ = PerformancePlot_Removal(
            vals={'Permuted Data Shapley': Per_DShap,
                  'TMC Data Shapley': TMC_Shap,
                  'Predicted Shapley': Pred_Shap},
            X_train=X_train.to_numpy(), y_train=y_train,
            X_test=X_test.to_numpy(),   y_test=y_test,
            a=a, b=b, ylim=ylim,
            order='ascending', random_iters=random_iters,
            save_path=asc_png
        )
        print('saved ➜', asc_png)

        # Magnitude Comparison and Linear Comparison
        tmc_vals  = np.asarray(TMC_Shap)
        per_vals  = np.asarray(Per_DShap)
        pred_vals = np.asarray(Pred_Shap)
        mean_tmc, mean_per, mean_pred = tmc_vals.mean(), per_vals.mean(), pred_vals.mean()
        N = len(per_vals); idx = np.arange(N); width = 0.25

        plt.figure(figsize=(10,6))
        plt.bar(idx, tmc_vals,  width, label='TMC Shapley',      edgecolor='k', alpha=0.7)
        plt.bar(idx, per_vals,  width, label='Permuted Shapley', edgecolor='k', alpha=0.7)
        plt.bar(idx+width, pred_vals, width, label='Predicted Shapley', edgecolor='k', alpha=0.7)
        plt.plot(idx, [mean_tmc]*N,  '--', linewidth=2, label=f'Avg TMC ({mean_tmc:.3f})')
        plt.plot(idx, [mean_per]*N,  '--', linewidth=2, label=f'Avg Permuted ({mean_per:.3f})')
        plt.plot(idx, [mean_pred]*N, '--', linewidth=2, label=f'Avg Pred ({mean_pred:.3f})')
        plt.xlabel('Sample Index'); plt.ylabel('Shapley Value')
        plt.title(f'Comparison of Shapley Values — {base_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
        bar_png = os.path.join(out_dir, f'{base_name}_ShapleyCompare.png')
        plt.savefig(bar_png, dpi=500); plt.close()
        print('saved ➜', bar_png)

        lo = min(per_vals.min(), tmc_vals.mean(), pred_vals.min())
        hi = max(per_vals.max(), tmc_vals.mean(), pred_vals.max())
        plt.figure(figsize=(8,8))
        plt.scatter(per_vals, tmc_vals,  s=120, alpha=0.6, edgecolors='k', linewidth=0.5, label='TMC vs Permuted')
        plt.scatter(per_vals, pred_vals, s=120, alpha=0.6, edgecolors='k', linewidth=0.5, label='Pred vs Permuted')
        plt.plot([lo-0.005, hi+0.005], [lo-0.005, hi+0.005], '--k', linewidth=1.5, label='y = x')
        plt.xlabel('Permuted Shapley Value'); plt.ylabel('Other Shapley Values')
        plt.title(f'Linear Comparison — {base_name}')
        plt.grid(True, linestyle='--', alpha=0.5); plt.legend(); plt.tight_layout()
        line_png = os.path.join(out_dir, f'{base_name}_LinearCompare.png')
        plt.savefig(line_png, dpi=500); plt.close()
        print('saved ➜', line_png)


def train_then_diagnose(BASE_TRAIN: str,
                        BASE_TEST: str,
                        OUT_DIR: str,
                        holdout_bases,
                        in_channels=10, hidden_channels=128, heads=4,
                        epochs=3000, lr=1e-5,
                        a=0.05, b=0.40, ylim=(40,105), random_iters=10):

    _ensure_dir(OUT_DIR)

    pairs = find_train_test_pairs(BASE_TRAIN)
    dataset_graphs = [load_and_preprocess(tr, te) for tr, te in pairs]

    gnn = GINShap(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads)
    loss_history = train(gnn, dataset_graphs, epochs=epochs, lr=lr)

    plt.figure(figsize=(6,4))
    plt.plot(np.log(loss_history), label='Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('log(Loss)')
    plt.title('Optimisation Curve(Log)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'Optimisation_Curve.png')); plt.close()

    diagnose_train_sets(gnn, base_train_dir=BASE_TRAIN, out_dir=OUT_DIR, graph_dir=OUT_DIR)
    diagnose_test_sets(gnn, base_test_dir=BASE_TEST, holdout_bases=holdout_bases, out_dir=OUT_DIR,
                       a=a, b=b, ylim=ylim, random_iters=random_iters)

    return gnn

# BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Train'
# BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test'
# OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250915/Trial02'
# HOLDOUTS   = ['MNIST_17_1', 'MNIST_17_2', 'MNIST_17_3']
#
# gnn = train_then_diagnose(
#     BASE_TRAIN=BASE_TRAIN,
#     BASE_TEST=BASE_TEST,
#     OUT_DIR=OUT_DIR,
#     holdout_bases=HOLDOUTS,
#     in_channels=10, hidden_channels=128, heads=4,
#     epochs=1500, lr=1e-6,
#     a=0.05, b=0.40, ylim=(40,105), random_iters=10
# )

PARAMS = dict(
    BASE_TRAIN = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialF_BC/Train',
    BASE_TEST  = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialF_BC/Test',
    OUT_DIR    = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/251005/Trial03',
    holdout_bases = ['bc_1', 'bc_2', 'bc_3'],

    # Pre-Processing
    d=10,
    C_List=(0.25, 0.5, 1, 2, 4),

    # model
    in_channels = 10,
    hidden_channels = 128,
    heads = 4,
    epochs = 1500,
    lr = 5e-5,

    # diagnostic plots
    a = 0.05,
    b = 0.40,
    ylim = (40, 105),
    random_iters = 10,
)
C_List = list(PARAMS["C_List"])

model_param = [
    "BASE_TRAIN","BASE_TEST","OUT_DIR","holdout_bases",
    "in_channels","hidden_channels","heads","epochs","lr",
    "a","b","ylim","random_iters"
]
param_setting = {k: PARAMS[k] for k in model_param}
gnn = train_then_diagnose(**param_setting)
