''' 250619 Experiment '''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
import numpy as np


# File paths
files = [
    '/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_5pts_CPU',
    '/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_10pts_CPU',
    '/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_15pts_CPU',
    '/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_20pts_CPU',
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


''' 250620 Experiment '''
df = pd.read_csv('/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_5pts_CPU')
X_raw = pd.get_dummies(df.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
D = pairwise_distances(X_raw, metric='euclidean')
sigma = np.mean(D)
A = np.exp(-(D ** 2) / (2 * (0.25 * sigma) ** 2))
A = np.maximum(A, A.T)
U, S, _ = randomized_svd(A, n_components = 3, random_state=42)
X_svd = U * np.sqrt(S)
X_svd_re = U@np.diagflat(S**0.5)

df2 = pd.read_csv('/tmp/pycharm_project_236/Outputs/250619/Trial01/Iris_01_10pts_CPU')
X_raw2 = pd.get_dummies(df2.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
D2 = pairwise_distances(X_raw2, metric='euclidean')
sigma2 = np.mean(D2)
A2 = np.exp(-(D2 ** 2) / (2 * (0.25 * sigma2) ** 2))
A2 = np.maximum(A2, A2.T)
U2, S2, _ = randomized_svd(A2, n_components = 10, random_state=42)
X_svd4 = U2 * np.sqrt(S2)
X_t4 = torch.tensor(X_svd4, dtype=torch.float)
X_t4 = filp_pe_sign(X_t4, method='l1')


df3 = pd.read_csv('/tmp/pycharm_project_236/Datasets/bc_1')
X_raw3 = pd.get_dummies(df3.drop(columns=['target','TrueDShap'])).astype(float).to_numpy()
D3 = pairwise_distances(X_raw3, metric='euclidean')
sigma3 = np.mean(D3)
A3 = np.exp(-(D3 ** 2) / (2 * (0.25 * sigma3) ** 2))
A3 = np.maximum(A3, A3.T)
U3, S3, _ = randomized_svd(A3, n_components = 10, random_state=42)
X_svd3 = U3 * np.sqrt(S3)
X_t3 = torch.tensor(X_svd3, dtype=torch.float)
X_t3 = filp_pe_sign(X_t3, method='l1')

FILE_PATH = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V2/Iris_12_large'
df = pd.read_csv(FILE_PATH)
df['TrueDShap'] = ''
df.to_csv(FILE_PATH, index=False)
print(f"Added empty 'TrueDShap' column and saved back to {FILE_PATH}")
