import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import traceback
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import combinations
import psutil
import torch
import torch.nn.functional as F
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)

# def train_subset(X_sub, y_sub, X_test, y_test):
#     X_sub = torch.tensor(X_sub.values, dtype = torch.float32, device=device)
#     y_sub = torch.tensor(y_sub.values, dtype = torch.float32, device=device)
#     X_test = torch.tensor(X_test.values, dtype = torch.float32, device=device)
#     y_test = torch.tensor(y_test.values, dtype = torch.float32, device=device)
#
#     model = TorchLogisticRegression(X_sub.shape[1]).to(device)
#     optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, lr = 1.0)
#
#     def closure():
#         optimizer.zero_grad()
#         loss = F.binary_cross_entropy(model(X_sub), y_sub)
#         loss.backward()
#         return loss
#
#     optimizer.step(closure)
#     model.eval()
#     with torch.no_grad():
#         preds = model(X_test).round()
#         acc = (preds == y_test).float().mean().item()
#     return acc

# def train_subset(X_sub, y_sub, X_test, y_test):
#     X_sub = torch.tensor(X_sub.values, dtype=torch.float32, device=device)
#     y_sub = torch.tensor(y_sub.values, dtype=torch.float32, device=device)
#     X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
#     y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)
#
#     model = TorchLogisticRegression(X_sub.shape[1]).to(device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # classic gradient descent
#     epochs = 100
#
#     for _ in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         output = model(X_sub).squeeze()
#         loss = F.binary_cross_entropy(output, y_sub)
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         preds = model(X_test).round()
#         acc = (preds == y_test).float().mean().item()
#     return acc


# iris = load_iris()
# X = pd.DataFrame(iris.data, columns = iris.feature_names)
# y = pd.Series(iris.target, name = 'target')
# X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
# iris_df = pd.concat([X_scaled, y], axis=1)
#
# iris_1 = iris_df[iris_df["target"] == 1].sample(n = 20, random_state = 333)
# iris_2 = iris_df[iris_df["target"] == 2].sample(n = 20, random_state = 333)
# iris_12 = pd.concat([iris_1, iris_2], axis = 0)
# iris_12['target'] = iris_12['target'].map({1:0, 2:1})
#
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_12.drop(columns = ['target']),
#     iris_12['target'],
#     test_size = 0.5,
#     stratify = iris_12['target'],
#     random_state = 111
# )


from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 20000, n_features = 1000, n_informative = 30, n_redundant = 10, n_classes = 2,
                           flip_y = 0.01, class_sep = 1.0, random_state = 909)
X = pd.DataFrame(X, columns = [f'f{i+1}' for i in range(X.shape[1])])
y = pd.Series(y, name = 'target')
print(X.shape, y.shape)
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y, random_state = 111)

# N = len(X_train)
# subset_size = 5
# all_subsets = list(combinations(range(N), subset_size))
# subset_keys = all_subsets[:2000]
#
# for batch_size in [100, 200, 500, 1000, 2000]:
#     print(f"Batch size: {batch_size}")
#     try:
#         t0 = time.time()
#         acc_list = []
#         for i in range(batch_size):
#             key = subset_keys[i]
#             X_sub = X_train.iloc[list(key)]
#             y_sub = y_train.iloc[list(key)]
#             acc = train_subset(X_sub, y_sub, X_test, y_test)
#             acc_list.append(acc)
#         print(f"Batch {batch_size} completed in {time.time() - t0:.2f} seconds")
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print("CUDA out of memory error. Reducing batch size.")
#         else:
#             print("Other error:", e)
#         traceback.print_exc()
#         break

'Runtime Comparison'
def train_sklearn_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, tol = 0.0, solver='saga', C = 1e12, verbose = 1)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return end - start
cpu_time = train_sklearn_model(X_train, y_train, X_test, y_test)
print(f"Logistic Regression in CPU: {cpu_time:.4f} seconds")

# def train_torch_model(X_train, y_train, X_test, y_test, device):
#     X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device)
#     y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
#     X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
#     y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)
#
#     model = TorchLogisticRegression(X_train.shape[1]).to(device)
#     optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100)
#
#     def closure():
#         optimizer.zero_grad()
#         loss = F.binary_cross_entropy(model(X_train), y_train)
#         loss.backward()
#         return loss
#
#     start = time.time()
#     optimizer.step(closure)
#     end = time.time()
#
#     return end - start

def train_torch_model(X_train, y_train, X_test, y_test, device, epochs = 1000, lr = 0.1):

    X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)

    model = TorchLogisticRegression(X_train.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    epochs = 1000

    start = time.time()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train).squeeze()
        loss = F.binary_cross_entropy(output, y_train)
        loss.backward()
        optimizer.step()
    end = time.time()
    return end - start

if torch.cuda.is_available():
    gpu_time = train_torch_model(X_train, y_train, X_test, y_test, device=torch.device("cuda"))
    print(f"GPU Time: {gpu_time:.4f} seconds")
else:
    print("CUDA not available. GPU test skipped.")
cpu_time = train_torch_model(X_train, y_train, X_test, y_test, device=torch.device("cpu"))
print(f"CPU Time: {cpu_time:.4f} seconds")