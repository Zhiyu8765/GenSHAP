''' 250619 Experiment '''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
import numpy as np

from Scripts.Discarded.GNN_DShap import PERF_PATH

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


'20250820 Experiment: Investigating convergence of TMC Shapley'
### Import packages
import matplotlib.pyplot as plt

' For Calculation of TMC Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#os.chdir('/tmp/pycharm_project_236/DataShapley')
sys.path.append(os.getcwd())
#sys.path.append('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')

### Package and Environments for TMC_Shapley
#%load_ext autoreload
#%autoreload 2
#import shap_utils
#importlib.reload(shap_utils)
import numpy as np
import pandas as pd
from DShap import DShap
from shap_utils import *

#%matplotlib inline
MEM_DIR = '../'


### Read in data
BASE = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test'
df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_1')
df_test = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_test_1')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','TrueDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
True_Shap = df_train['TrueDShap'].to_numpy()
X_test = pd.get_dummies(df_test.drop(
                columns=['target'])).to_numpy()
y_test = df_test['target'].to_numpy()

### Compute TMC Shapley values
directory = './temp'
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
shapley_model = DShap(
    X_train, y_train, X_test, y_test, num_test = 20,
    sources=None, sample_weight=None, model_family='logistic',
    metric='accuracy', overwrite=True, directory=directory,
    seed=250820, max_iter=100, err=0.005, tol=1e-5)
shapley_model.run(save_every=100, err=0.005, g_run=True, loo_run=True)
TMC_Shap = shapley_model.vals_tmc
print("Total TMC-Shapley iterations (t):", shapley_model.mem_tmc.shape[0])

from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
PERF_PATH = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250820/TMC_Tune/Trial04/MNIST_17_1_Low'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,105),
    order = 'ascending', random_iters = 10,
    save_path = PERF_PATH
)

'Distribution Comparison'
GRAPH_DIR = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250820/TMC_Tune/Trial04/'
true_vals = np.asarray(True_Shap)
tmc_vals  = np.asarray(TMC_Shap)
mean_true = true_vals.mean()
mean_tmc  = tmc_vals.mean()
N = len(true_vals)
indices = np.arange(N)
width = 0.25
plt.figure(figsize=(10,6))
plt.bar(indices - width, true_vals,  width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
plt.bar(indices, tmc_vals, width, label='TMC Shapley', color='tab:orange',edgecolor='k', alpha=0.7)
plt.plot(indices, [mean_true]*N, linestyle='--', linewidth=2, label=f'Average True Shapley Values ({mean_true:.3f})',   color='tab:blue')
plt.plot(indices, [mean_tmc]*N, linestyle='--', linewidth=2, label=f'Average TMC Shapley Values ({mean_tmc:.3f})',     color='tab:orange')
plt.xlabel('Sample Index', fontsize=12, weight='bold')
plt.ylabel('Shapley Value', fontsize=12, weight='bold')
plt.title('Comparison of True, TMC & Predicted Shapley Values', fontsize=14, weight='bold')
plt.xticks(indices)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, 'Shapley_compare_MNIST_17_1.png')
plt.savefig(out_path, dpi=500)
plt.show()


'Linear Plot'
true_vals = np.asarray(True_Shap)
tmc_vals  = np.asarray(TMC_Shap)

plt.figure(figsize=(8,8))
plt.scatter(true_vals, tmc_vals,
            s=200, color='tab:orange', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='TMC Shapley')
all_min = min(true_vals.min(), tmc_vals.min())
all_max = max(true_vals.max(), tmc_vals.max())
plt.plot([all_min-0.02, all_max+0.02], [all_min-0.02, all_max+0.02],
         '--k', linewidth=1.5, label='y = x')

plt.xlabel('True TMC Shapley Values', fontsize=10, weight='bold')
plt.ylabel('Predicted TMC Shapley Values', fontsize=10, weight='bold')
plt.title('True vs Predicted & TMC Shapley Values', fontsize=12, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

out_path = os.path.join(GRAPH_DIR, 'LinearCompare_MNIST_17_1.png')
plt.savefig(out_path, dpi=500)
plt.tight_layout()
plt.show()


'20250831 Experiment: Investigating convergence of Permutation-Based Data Shapley'
### Import packages
import matplotlib.pyplot as plt

' For Calculation of Permutation Based Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
#os.chdir('/tmp/pycharm_project_236/DataShapley')
sys.path.append(os.getcwd())
#sys.path.append('/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')

### Package and Environments for TMC_Shapley
#%load_ext autoreload
#%autoreload 2
#import shap_utils
#importlib.reload(shap_utils)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#%matplotlib inline
MEM_DIR = '../'


### Read in data
BASE = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test'
df_train = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_1')
df_test = pd.read_csv('/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/10n15p20/Test/MNIST_17_test_1')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','TrueDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
True_Shap = df_train['TrueDShap'].to_numpy()
X_test = pd.get_dummies(df_test.drop(
                columns=['target'])).to_numpy()
y_test = df_test['target'].to_numpy()

### Compute TMC Shapley values
Per_Shap = DataShapley_permutation(X_train, y_train, X_test, y_test, n_permutations=5000,
    model_builder=lambda: LogisticRegression(max_iter=100, n_jobs=1),
    random_state=0, progress=True)

from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
PERF_PATH = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250831/Per_Shap/Trial05/MNIST_17_1_Low'
removal_percentages, results = PerformancePlot_Removal(
    vals={'True Data Shapley': True_Shap,
          'Permutation-Based Data Shapley': Per_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (40,105),
    order = 'ascending', random_iters = 10,
    save_path = PERF_PATH
)

'Distribution Comparison'
GRAPH_DIR = '/Users/cuhksz-ml-01/Library/CloudStorage/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250831/Per_Shap/Trial05/'
true_vals = np.asarray(True_Shap)
per_vals  = np.asarray(Per_Shap)
mean_true = true_vals.mean()
mean_per  = per_vals.mean()
N = len(true_vals)
indices = np.arange(N)
width = 0.25
plt.figure(figsize=(10,6))
plt.bar(indices - width, true_vals,  width, label='True Shapley', color='tab:blue',  edgecolor='k', alpha=0.7)
plt.bar(indices, per_vals, width, label='Permutation-Based Shapley', color='tab:orange',edgecolor='k', alpha=0.7)
plt.plot(indices, [mean_true]*N, linestyle='--', linewidth=2, label=f'Average True Shapley Values ({mean_true:.3f})',   color='tab:blue')
plt.plot(indices, [mean_per]*N, linestyle='--', linewidth=2, label=f'Average Permutation-Based Shapley Values ({mean_per:.3f})',     color='tab:orange')
plt.xlabel('Sample Index', fontsize=12, weight='bold')
plt.ylabel('Shapley Value', fontsize=12, weight='bold')
plt.title('Comparison of True, Permutation-Based Shapley Values', fontsize=14, weight='bold')
plt.xticks(indices)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, 'Shapley_compare_MNIST_17_1.png')
plt.savefig(out_path, dpi=500)
plt.show()


'Linear Plot'
true_vals = np.asarray(True_Shap)
per_vals  = np.asarray(Per_Shap)

plt.figure(figsize=(8,8))
plt.scatter(true_vals, per_vals,
            s=200, color='tab:orange', alpha=0.6, edgecolors='k',
            linewidth=0.5, label='Permutation-Based Shapley')
all_min = min(true_vals.min(), per_vals.min())
all_max = max(true_vals.max(), per_vals.max())
plt.plot([all_min-0.02, all_max+0.02], [all_min-0.02, all_max+0.02],
         '--k', linewidth=1.5, label='y = x')

plt.xlabel('True TMC Shapley Values', fontsize=10, weight='bold')
plt.ylabel('Permutation-Based Shapley Values', fontsize=10, weight='bold')
plt.title('True vs Predicted & Permutation-Based Shapley Values', fontsize=12, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

out_path = os.path.join(GRAPH_DIR, 'LinearCompare_MNIST_17_1.png')
plt.savefig(out_path, dpi=500)
plt.tight_layout()
plt.show()



'20250908 Removal plots of Test Sets'
df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_test_1')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
df_tmc = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_TMC_1')
df_train = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/MNIST_17_1')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','PerDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
# True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
Per_DShap = df_train['PerDShap'].to_numpy()
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialB_MNIST/Test/', 'MNIST_17_1_Low.png')
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={ 'Permuted Data Shapley': Per_DShap,
        # 'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (80,105),
    order = 'ascending', random_iters = 10,
    save_path = PERF_PATH
)

df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialA_Iris/Test/Iris_02_test_1')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
df_tmc = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialA_Iris/Test/Iris_02_TMC_1')
df_train = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialA_Iris/Test/Iris_02_1')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','PerDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
# True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
Per_DShap = df_train['PerDShap'].to_numpy()
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialA_Iris/Test/', 'Iris_02_1_High.png')
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={ 'Permuted Data Shapley': Per_DShap,
        # 'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (80,105),
    order = 'descending', random_iters = 10,
    save_path = PERF_PATH
)

df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialC_Iris/Test/Iris_02_test_1')
X_test = pd.get_dummies(df_test.drop(columns = ['target'])).to_numpy()
y_test = df_test['target'].to_numpy()
df_tmc = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialC_Iris/Test/Iris_02_TMC_1')
df_train = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialC_Iris/Test/Iris_02_1')
X_train = pd.get_dummies(df_train.drop(
                columns=['target','PerDShap'])).to_numpy()
y_train = df_train['target'].to_numpy()
# True_Shap = df_train['TrueDShap'].to_numpy()
TMC_Shap = df_tmc['TMC_Shap'].to_numpy()
Per_DShap = df_train['PerDShap'].to_numpy()
PERF_PATH = os.path.join('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/Per_TrainnTest/TrialC_Iris/Test/', 'Iris_02_1_Low.png')
from Scripts.Additional_Scripts.PerformancePlots_4lines import PerformancePlot_Removal
removal_percentages, results = PerformancePlot_Removal(
    vals={ 'Permuted Data Shapley': Per_DShap,
        # 'True Data Shapley': True_Shap,
          'TMC Data Shapley': TMC_Shap},
    X_train = X_train, y_train = y_train,
    X_test = X_test, y_test = y_test,
    a = 0.05, b = 0.4, ylim = (80,105),
    order = 'ascending', random_iters = 10,
    save_path = PERF_PATH
)



'20250916 Experiment: Investigating impact of changing 1 point in training set on data Shapley values'
df_train = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/RE/Iris_12_train_1')
df_test = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/RE/Iris_12_test_1')
X_train = df_train.drop(columns = ['target'])
y_train = df_train['target']
X_test = df_test.drop(columns = ['target'])
y_test = df_test['target']
save_dir = '/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/RE'
## Run all 3 methods' algorithm in other files
a = 1
b = 2
i = 1
D_Shap = DataShapley(X_train, y_train, X_test, y_test)
Iris_train = X_train.copy()
Iris_train['target'] = y_train.values
Iris_train['TrueDShap'] = D_Shap
Iris_train.to_csv(f'{save_dir}/Iris_{a}{b}_{i}', index=False)

phi = Per_DShap(X_train, y_train, X_test, y_test, n_permutations=5000,
                model_builder=lambda: LogisticRegression(max_iter=100, n_jobs=1),
                random_state=0, progress=True)
Iris_per = X_train.copy()
Iris_per['target'] = y_train.values
Iris_per['PerDShap'] = phi
Iris_per.to_csv(f'{save_dir}/Iris_{a}{b}_per_{i}', index=False)

num_test = 15
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
directory = './temp'
shapley_model = DShap(X_train.values, y_train.values,
                      X_test.values, y_test.values,
                      num_test, sources=None,
                      sample_weight=None, model_family=model,
                      metric=metric, overwrite=True,
                      directory=directory, seed=250625 + i,
                      max_iter=100, tol=1e-5)
shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
Iris_TMC = Iris_train.copy()
Iris_TMC['TMC_Shap'] = shapley_model.vals_tmc
Iris_TMC.to_csv(f'{save_dir}/Iris_{a}{b}_TMC_{i}', index=False)

'Read original values'
df_DShap0 = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/Original/Iris_12_1')
df_PShap0 = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/Original/Iris_12_per_1')
df_TMC0 = pd.read_csv('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/Original/Iris_12_TMC_1')
DShap0 = df_DShap0['TrueDShap'].to_numpy()
PShap0 = df_PShap0['PerDShap'].to_numpy()
TMC0 = df_TMC0['TMC_Shap'].to_numpy()

'Generate Linear Comparison Plots'
methods = ['True', 'Per', 'TMC']
for method in methods:
    df_new = pd.read_csv(f'/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/RE/Iris_12_{method}_1')
    if method == 'True':
        new_vals = df_new['TrueDShap'].to_numpy()
        orig_vals = DShap0
    elif method == 'Per':
        new_vals = df_new['PerDShap'].to_numpy()
        orig_vals = PShap0
    else:
        new_vals = df_new['TMC_Shap'].to_numpy()
        orig_vals = TMC0

    plt.figure(figsize=(8,8))
    plt.scatter(orig_vals, new_vals,
                s=200, color='tab:orange', alpha=0.6, edgecolors='k',
                linewidth=0.5, label=f'{method} after changing 1 point')
    all_min = min(orig_vals.min(), new_vals.min())
    all_max = max(orig_vals.max(), new_vals.max())
    plt.plot([all_min-0.02, all_max+0.02], [all_min-0.02, all_max+0.02],
             '--k', linewidth=1.5, label='y = x')

    plt.xlabel('Original Shapley Values', fontsize=10, weight='bold')
    plt.ylabel(f'New {method} Values', fontsize=10, weight='bold')
    plt.title(f'Original vs New {method} Values after Changing 1 Point', fontsize=12, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    out_path = f'/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250916/Trial01/RE/LinearCompare_Iris_12_{method}_1.png'
    plt.savefig(out_path, dpi=500)
    plt.tight_layout()
    plt.show()



# pyDVL ≥ 0.10.0 multi-method valuation on your MNIST CSVs

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ---------- pyDVL imports (with stable module paths) ----------
from pydvl.valuation import (
    Dataset, SupervisedScorer, ModelUtility,
    ShapleyValuation, BetaShapleyValuation, DeltaShapleyValuation,
    GroupTestingShapleyValuation
)

# Class-wise Shapley pieces
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility
from pydvl.valuation.samplers.classwise import ClasswiseSampler

# Samplers
from pydvl.valuation.samplers import PermutationSampler, OwenSampler, GridOwenStrategy
from pydvl.valuation.samplers.powerset import UniformSampler, NoIndexIteration
from pydvl.valuation.samplers.truncation import RelativeTruncation

# Stopping criteria
from pydvl.valuation.stopping import (
    RankCorrelation, MinUpdates, MaxSamples, History, NoStopping
)

# KNN Shapley (exact)
from pydvl.valuation.methods.knn_shapley import KNNShapleyValuation

# ---------- Paths & columns ----------
TRAIN_CSV = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251031/Trial01/MNIST_369xx_20.csv"
TEST_CSV  = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251031/Trial01/MNIST_369xx_test_20.csv"
OUT_DIR   = os.path.dirname(TRAIN_CSV)
LABEL_COL = "target"
PRED_CSV  = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251031/Trial01/MNIST_369xx_20_Pred.csv"
TMC_CSV   = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs_II/251031/Trial01/MNIST_369xx_TMC_20.csv"
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

assert LABEL_COL in df_train.columns and LABEL_COL in df_test.columns, "Label column not found."

drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]

X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()

X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()

# --- Create pyDVL Datasets (constructor, not from_arrays) ---
train_ds = Dataset(X_train, y_train)
test_ds  = Dataset(X_test,  y_test)

# ---------- Base models ----------
# (A) Logistic regression for model-based methods
lr = LogisticRegression(max_iter=200)
scorer = SupervisedScorer("accuracy", test_ds, default=0.0, range=(0.0, 1.0))
utility = ModelUtility(lr, scorer)

# (B) KNN for KNN-Shapley
knn = KNeighborsClassifier(n_neighbors=3)

# ---------- Common stoppers ----------
# Reasonable small-ish budgets; tune up if you want higher precision.
shapley_stopping   = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
beta_stopping      = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
delta_stopping     = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)
group_stopping     = MinUpdates(5000) | RankCorrelation(rtol=1e-3, burn_in=200)

# ---------- 1) Standard Shapley (permutation) ----------
perm_sampler = PermutationSampler(truncation=RelativeTruncation(rtol=0.01, burn_in_fraction=0.3), seed=42)
val_shapley  = ShapleyValuation(utility, perm_sampler, shapley_stopping, progress=True)
val_shapley.fit(train_ds)
res_shapley = val_shapley.result.values

# ---------- 2) Beta Shapley ----------
val_beta = BetaShapleyValuation(utility, perm_sampler, beta_stopping, alpha=1, beta=16, progress=True)
val_beta.fit(train_ds)
res_beta = val_beta.result.values

# # ---------- 4) Group Testing Shapley ----------
# import inspect
# from pydvl.valuation.methods.gt_shapley import GroupTestingShapleyValuation
# from pydvl.valuation.samplers import PermutationSampler
#
# perm_sampler = PermutationSampler()  # or keep your configured one
# N_SAMPLES    = 2000
# BATCH        = 64
#
# import inspect
# from pydvl.valuation.methods.gt_shapley import GroupTestingShapleyValuation
#
# def make_group_testing_vrobust(utility, sampler, is_done,
#                                n_samples=2000, batch_size=64,
#                                epsilon=1e-3, progress=True):
#     """
#     Build GroupTestingShapleyValuation across pyDVL variants:
#     - pass only kwargs present in the __init__ signature
#     - ensure epsilon (tolerance) is provided when required
#     - avoid positional args to prevent mis-binding
#     - attach sampler afterward if ctor doesn't accept it
#     """
#     sig = inspect.signature(GroupTestingShapleyValuation)
#     allowed = set(sig.parameters.keys())
#
#     kwargs = {}
#     if "utility"        in allowed: kwargs["utility"]        = utility
#     if "sampler"        in allowed: kwargs["sampler"]        = sampler
#     if "is_done"        in allowed: kwargs["is_done"]        = is_done
#     if "n_samples"      in allowed: kwargs["n_samples"]      = int(n_samples)
#     if "batch_size"     in allowed: kwargs["batch_size"]     = int(batch_size)
#     if "epsilon"        in allowed: kwargs["epsilon"]        = float(epsilon)
#     if "tol"            in allowed and "epsilon" not in allowed:
#         kwargs["tol"] = float(epsilon)                       # some variants use 'tol'
#     if "progress"       in allowed: kwargs["progress"]       = bool(progress)
#     if "solver_options" in allowed: kwargs["solver_options"] = {}  # ensure dict
#
#     val = GroupTestingShapleyValuation(**kwargs)
#
#     # if ctor didn't take 'sampler', attach it when possible
#     if getattr(val, "sampler", None) is None:
#         try: val.sampler = sampler
#         except Exception: pass
#     return val
#
# # ---- build & run ----
# val_group = make_group_testing_vrobust(
#     utility=utility,
#     sampler=perm_sampler,
#     is_done=group_stopping,
#     n_samples=N_SAMPLES,
#     batch_size=BATCH,
#     progress=True,
# )
# val_group.fit(train_ds)
# res_group = val_group.result.values
# print("Group-Testing Shapley OK:", res_group.shape)

# # ---------- 5) Owen-Shapley (antithetic Owen or grid Owen are both valid) ----------
# owen_sampler = OwenSampler(outer_sampling_strategy=GridOwenStrategy(n_samples_outer=200))
# owen_stop    = RankCorrelation(rtol=1e-3, burn_in=100)
# val_owen     = ShapleyValuation(utility, owen_sampler, owen_stop, progress=True)
# val_owen.fit(train_ds)
# res_owen = val_owen.result.values
#
# # ---------- 6) Class-wise Shapley ----------
# # per docs: in-class = permutations (finite), out-of-class = powerset with NoIndexIteration
# cws_scorer  = ClasswiseSupervisedScorer("accuracy", test_ds)  # defaults follow paper (f(x)=x, g(x)=exp(x))
# cws_utility = ClasswiseModelUtility(lr, cws_scorer)
# cws_sampler = ClasswiseSampler(
#     in_class=PermutationSampler(truncation=RelativeTruncation(rtol=0.01, burn_in_fraction=0.3), seed=42),
#     out_of_class=UniformSampler(index_iteration=NoIndexIteration),
#     max_in_class_samples=1,
# )
# # 500 permutations per label as in the paper; feel free to bump this
# n_labels  = len(np.unique(y_train))
# cws_stop  = MaxSamples(cws_sampler, 500 * n_labels) | History(n_steps=5000)
# val_cws   = None
# try:
#     # New import path (already used above) ensures no ModuleNotFoundError
#     from pydvl.valuation.methods.classwise_shapley import ClasswiseShapleyValuation
#     val_cws = ClasswiseShapleyValuation(cws_utility, cws_sampler, cws_stop, normalize_values=True, progress=True)
#     val_cws.fit(train_ds)
#     res_cws = val_cws.result.values
# except Exception as e:
#     warnings.warn(f"Class-wise Shapley failed to run: {e}")
#     res_cws = np.full(len(train_ds), np.nan)

# ---------- 7) KNN-Shapley (exact, O(n log n) per test point) ----------
val_knn = KNNShapleyValuation(knn, test_ds, progress=True)
val_knn.fit(train_ds)
res_knn = val_knn.result.values

# ---------- Collect & save ----------
out = pd.DataFrame({
    "Index": df_train.index,
    LABEL_COL: df_train[LABEL_COL],
    # "pyDVL_Shapley": res_shapley,
    "pyDVL_BetaShapley": res_beta,
    # "pyDVL_DeltaShapley": res_delta,
    # "pyDVL_GroupTesting": res_group,
    # "pyDVL_OwenShapley": res_owen,
    # "pyDVL_Classwise": res_cws,
    "pyDVL_KNNShapley": res_knn,
})

# Join your existing columns if present
def _safe_merge(base: pd.DataFrame, path: str, suffix: str) -> pd.DataFrame:
    if os.path.exists(path):
        extra = pd.read_csv(path)
        # Try common join keys: 'Index' then fall back to positional if needed
        if "Index" in extra.columns:
            return base.merge(extra, on="Index", how="left", suffixes=("", suffix))
        else:
            extra = extra.copy()
            extra["Index"] = extra.index
            return base.merge(extra, on="Index", how="left", suffixes=("", suffix))
    return base

out = _safe_merge(out, PRED_CSV,  "_pred")
out = _safe_merge(out, TMC_CSV,   "_tmc")

OUT_CSV = os.path.join(OUT_DIR, "MNIST_369xx_20_pyDVL_AllValues.csv")
out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)


# --- Linear comparison: Pred/TMC/Beta/KNN vs Permuted (PerDShap) ---
# === Linear comparison: Pred/TMC/Beta/KNN vs PerDShap ===
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv(OUT_CSV)

# Columns we will use (PerDShap + Pred_Shap are guaranteed by you)
PERM_COL = "PerDShap"
PRED_COL = "Pred_Shap"
BETA_COL = "pyDVL_BetaShapley"
KNN_COL  = "pyDVL_KNNShapley"

# TMC column: try common names then fall back to first col containing 'tmc' and 'shap'
if "TMC_Shap" in df.columns:
    TMC_COL = "TMC_Shap"
elif "TMC_Shapley" in df.columns:
    TMC_COL = "TMC_Shapley"
else:
    tmc_guess = [c for c in df.columns if ("tmc" in c.lower() and "shap" in c.lower())]
    if not tmc_guess:
        raise ValueError("Couldn't find a TMC column; please add one named 'TMC_Shap' (or similar).")
    TMC_COL = tmc_guess[0]

# Label column
if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in OUT_CSV.")
y_vals = df[LABEL_COL].to_numpy()

# Values
per_vals  = df[PERM_COL].to_numpy()
pred_vals = df[PRED_COL].to_numpy()
tmc_vals  = df[TMC_COL].to_numpy()
beta_vals = df[BETA_COL].to_numpy()
knn_vals  = df[KNN_COL].to_numpy()

# Axis limits
lo = np.nanmin([per_vals.min(), pred_vals.min(), tmc_vals.min(), beta_vals.min(), knn_vals.min()])
hi = np.nanmax([per_vals.max(), pred_vals.max(), tmc_vals.max(), beta_vals.max(), knn_vals.max()])
pad = 0.001 * (hi - lo + 1e-12)

# Styles
COLORS  = {"pred":"#1f77b4","tmc":"#ff7f0e","beta":"#2ca02c","knn":"#d62728"}
MARKERS = ['o','s','^','D','P','X','v','<','>','*']
classes = np.unique(y_vals)
styles  = {c: dict(marker=MARKERS[i % len(MARKERS)], s=20, linewidths=1.2) for i, c in enumerate(classes)}

# Plot
fig, ax = plt.subplots(figsize=(8.8, 8.8))
for c in classes:
    m  = (y_vals == c)
    st = styles[c].copy()
    ax.scatter(per_vals[m], pred_vals[m], **st, facecolors='none', edgecolors=COLORS["pred"])
    ax.scatter(per_vals[m], tmc_vals[m],  **st, facecolors='none', edgecolors=COLORS["tmc"])
    ax.scatter(per_vals[m], beta_vals[m], **st, facecolors='none', edgecolors=COLORS["beta"])
    ax.scatter(per_vals[m], knn_vals[m],  **st, facecolors='none', edgecolors=COLORS["knn"])

ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--k', linewidth=1.5)
ax.set_xlabel(f'Permuted Shapley Value ({PERM_COL})')
ax.set_ylabel('Other Shapley Values')
ax.set_title('Linear Comparison — MNIST_369xx_20')
ax.grid(True, linestyle='--', alpha=0.5)

# Legends
series_handles = [
    Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["pred"], markersize=8, label='Predicted Shapley'),
    Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["tmc"],  markersize=8, label='TMC Shapley'),
    Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["beta"], markersize=8, label='Beta Shapley'),
    Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor=COLORS["knn"],  markersize=8, label='KNN Shapley'),
]
legend_series = ax.legend(handles=series_handles, title="Series (color)", loc='upper left', fontsize=9)
ax.add_artist(legend_series)

class_handles = [
    Line2D([0],[0], marker=styles[c]['marker'], linestyle='None',
           markerfacecolor='none', markeredgecolor='k', markersize=8, label=f'class {int(c)}')
    for i, c in enumerate(classes)
]
legend_classes = ax.legend(handles=class_handles, title="Class (marker)", loc='lower right', fontsize=9)
ax.add_artist(legend_classes)

plt.tight_layout()
line_png = os.path.join(OUT_DIR, "MNIST_369xx_20_LinearCompare.png")
plt.savefig(line_png, dpi=500)
plt.close()
print("Saved linear comparison to:", line_png)


'''
260112. Histogram of Shapley values
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


'1. Characters'
CSV_PATH = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/characters_048_1'
df = pd.read_csv(CSV_PATH)
x = np.arange(len(df))
y = df['PerDShap'].to_numpy()

## Histogram
plt.figure(figsize=(12, 8))
plt.hist(y, bins=50)
plt.xlabel("PerDShap Values")
plt.ylabel("Count")
plt.title("Histogram of PerDShap")
plt.tight_layout()
plt.show()

## Scatter plot
X = df.drop(columns=["PerDShap", "target"])
y = df["target"].to_numpy()                 # class labels (should be 3 unique values)
per = df["PerDShap"].to_numpy(dtype=float)

# PCA
Xz = StandardScaler().fit_transform(X.to_numpy(dtype=float))
Z = PCA(n_components=2, random_state=0).fit_transform(Xz)

# sizes from |PerDShap|
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)
s_min, s_max = 10.0, 250.0
sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float) if np.allclose(mag.max(), mag.min()) \
    else s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---- force exactly K discrete colors ----
classes = np.unique(y)
K = len(classes)
print("Unique classes:", classes, "K =", K)

# map labels -> 0..K-1
class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

# discrete colormap + boundary norm (no interpolation)
cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

# legend (exactly K entries)
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title="target", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()

## t-SNE
TARGET_COL = "target"
PERDSHAP_COL = "PerDShap"
X = df.drop(columns=[PERDSHAP_COL, TARGET_COL]).to_numpy(dtype=float)
y = df[TARGET_COL].to_numpy()
per = df[PERDSHAP_COL].to_numpy(dtype=float)

# ---------------------------
# Size mapping: |PerDShap| -> marker area (robust)
# ---------------------------
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)

s_min, s_max = 10.0, 250.0
if np.allclose(mag.max(), mag.min()):
    sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float)
else:
    sizes = s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---------------------------
# Standardize then t-SNE to 2D
# ---------------------------
Xz = StandardScaler().fit_transform(X)
Z = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    init="pca",
    random_state=0
).fit_transform(Xz)

# ---------------------------
# Discrete colors: target -> {0,...,K-1}
# ---------------------------
classes = np.unique(y)
K = len(classes)

class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title=TARGET_COL, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()

'2. Iris'
CSV_PATH = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/Iris_1'
df = pd.read_csv(CSV_PATH)
x = np.arange(len(df))
y = df['PerDShap'].to_numpy()

plt.figure(figsize=(12, 8))
plt.hist(y, bins=50)
plt.xlabel("PerDShap Values")
plt.ylabel("Count")
plt.title("Histogram of PerDShap")
plt.tight_layout()
plt.show()

X = df.drop(columns=["PerDShap", "target"])
y = df["target"].to_numpy()                 # class labels (should be 3 unique values)
per = df["PerDShap"].to_numpy(dtype=float)

# PCA
Xz = StandardScaler().fit_transform(X.to_numpy(dtype=float))
Z = PCA(n_components=2, random_state=0).fit_transform(Xz)

# sizes from |PerDShap|
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)
s_min, s_max = 10.0, 250.0
sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float) if np.allclose(mag.max(), mag.min()) \
    else s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---- force exactly K discrete colors ----
classes = np.unique(y)
K = len(classes)
print("Unique classes:", classes, "K =", K)

# map labels -> 0..K-1
class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

# discrete colormap + boundary norm (no interpolation)
cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

# legend (exactly K entries)
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title="target", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()

## t-SNE
TARGET_COL = "target"
PERDSHAP_COL = "PerDShap"
X = df.drop(columns=[PERDSHAP_COL, TARGET_COL]).to_numpy(dtype=float)
y = df[TARGET_COL].to_numpy()
per = df[PERDSHAP_COL].to_numpy(dtype=float)

# ---------------------------
# Size mapping: |PerDShap| -> marker area (robust)
# ---------------------------
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)

s_min, s_max = 10.0, 250.0
if np.allclose(mag.max(), mag.min()):
    sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float)
else:
    sizes = s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---------------------------
# Standardize then t-SNE to 2D
# ---------------------------
Xz = StandardScaler().fit_transform(X)
Z = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0
).fit_transform(Xz)

# ---------------------------
# Discrete colors: target -> {0,...,K-1}
# ---------------------------
classes = np.unique(y)
K = len(classes)

class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title=TARGET_COL, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()



'3. Vehicle'
CSV_PATH = '/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Test/Vehicle_01_1'
df = pd.read_csv(CSV_PATH)
x = np.arange(len(df))
y = df['PerDShap'].to_numpy()

plt.figure(figsize=(12, 8))
plt.hist(y, bins=50)
plt.xlabel("PerDShap Values")
plt.ylabel("Count")
plt.title("Histogram of PerDShap")
plt.tight_layout()
plt.show()

X = df.drop(columns=["PerDShap", "target"])
y = df["target"].to_numpy()                 # class labels (should be 3 unique values)
per = df["PerDShap"].to_numpy(dtype=float)

# PCA
Xz = StandardScaler().fit_transform(X.to_numpy(dtype=float))
Z = PCA(n_components=2, random_state=0).fit_transform(Xz)

# sizes from |PerDShap|
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)
s_min, s_max = 10.0, 250.0
sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float) if np.allclose(mag.max(), mag.min()) \
    else s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---- force exactly K discrete colors ----
classes = np.unique(y)
K = len(classes)
print("Unique classes:", classes, "K =", K)

# map labels -> 0..K-1
class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

# discrete colormap + boundary norm (no interpolation)
cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

# legend (exactly K entries)
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title="target", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()

## t-SNE
TARGET_COL = "target"
PERDSHAP_COL = "PerDShap"
X = df.drop(columns=[PERDSHAP_COL, TARGET_COL]).to_numpy(dtype=float)
y = df[TARGET_COL].to_numpy()
per = df[PERDSHAP_COL].to_numpy(dtype=float)

# ---------------------------
# Size mapping: |PerDShap| -> marker area (robust)
# ---------------------------
mag = np.abs(per)
q1, q99 = np.quantile(mag, [0.01, 0.99])
mag = np.clip(mag, q1, q99)

s_min, s_max = 10.0, 250.0
if np.allclose(mag.max(), mag.min()):
    sizes = np.full_like(mag, (s_min + s_max) / 2.0, dtype=float)
else:
    sizes = s_min + (mag - mag.min()) * (s_max - s_min) / (mag.max() - mag.min())

# ---------------------------
# Standardize then t-SNE to 2D
# ---------------------------
Xz = StandardScaler().fit_transform(X)
Z = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0
).fit_transform(Xz)

# ---------------------------
# Discrete colors: target -> {0,...,K-1}
# ---------------------------
classes = np.unique(y)
K = len(classes)

class_to_int = {c: i for i, c in enumerate(classes)}
c_int = np.array([class_to_int[v] for v in y], dtype=int)

cmap = ListedColormap(plt.get_cmap("tab10").colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=c_int, s=sizes, alpha=0.75, cmap=cmap, norm=norm)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE scatter: color=target (discrete), size=|PerDShap|")
plt.tight_layout()

handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
               label=str(cls), markerfacecolor=cmap(i), markeredgecolor="none")
    for i, cls in enumerate(classes)
]
plt.legend(handles=handles, title=TARGET_COL, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.show()



########################################################################################################################
########################################################################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 1) Removal + Addition plots (4 plots per holdout set), each plot has 6 lines:
    - Pred_A, Pred_B, Pred_C, Pred_D (from column Pred_Shap in the corresponding files)
    - PerDShap (from column PerDShap)
    - Random baseline (avg over random_iters)
Task 2) Noisy label detection with 10% manual flips and k-means rule:
    - KMeans k=2 on 1D values
    - noisy if value < min(cluster_centers)
    - report F1 for A,B,C,D,PerDShap

Expected files in comparison_dir:
    characters_048_1_Pred_A.(csv|parquet|pkl)
    characters_048_1_Pred_B.(csv|parquet|pkl)
    ...
and similarly for the 6 holdout bases you listed.

Assumptions about each file:
    - It contains training-set rows (same row order across A/B/C/D for the same base)
    - Columns include:
        * 'target' (binary/int labels)
        * 'Pred_Shap' (predicted data value for that variant)
        * 'PerDShap' (baseline valuation)
    - Feature columns are numeric columns excluding known non-feature columns.
"""

import os
import glob
import argparse
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---- Import your helper removal-plot function (attached) ----
# Make sure PerformancePlots_4lines.py is in the same folder as this script,
# or provide --helper_path to point to it.
import importlib.util


# ----------------------------
# Utilities: IO + feature pick
# ----------------------------
SUPPORTED_EXTS = (".csv", ".parquet", ".pkl", ".pickle")


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported extension: {ext} for {path}")


def find_file(comparison_dir: str, base: str, variant: str) -> str:
    """
    Find file like: {base}_Pred_{variant}.* among supported extensions.
    """
    pattern = os.path.join(comparison_dir, f"{base}_Pred_{variant}*")
    candidates = []
    for p in glob.glob(pattern):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
            candidates.append(p)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No file found for pattern {pattern} with exts {SUPPORTED_EXTS}")
    if len(candidates) > 1:
        # pick the first deterministically
        candidates = sorted(candidates)
    return candidates[0]


def infer_feature_columns(df: pd.DataFrame,
                          exclude: List[str] = None) -> List[str]:
    """
    Infer feature columns as numeric columns excluding label/value columns.
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)

    # keep only numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in exclude_set]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after exclusion. "
                         "Please edit infer_feature_columns() to match your schema.")
    return feat_cols


# -----------------------------------------
# Addition curve (analogous to removal curve)
# -----------------------------------------
def performance_curve_addition(vals: np.ndarray,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               a: float,
                               b: float,
                               order: str = "descending") -> Tuple[List[float], np.ndarray]:
    """
    Build an addition curve by starting from empty and adding points by value order.
    order='descending' => add highest first; order='ascending' => add lowest first.
    Returns (addition_percentages, curve_accuracy_array).
    """
    n = len(X_train)
    total_add = int(np.floor(b * n))
    step_add  = max(1, int(np.floor(a * n)))
    add_counts = list(range(0, total_add + 1, step_add))
    add_percentages = [100 * r / n for r in add_counts]

    order_idx = np.argsort(vals)
    if order != "ascending":
        order_idx = order_idx[::-1]

    curve = []
    for r in add_counts:
        if r == 0:
            # no data: majority dummy (or random). We'll do majority=most common in y_train.
            maj = int(pd.Series(y_train).mode().iloc[0])
            dummy = np.full_like(y_test, maj)
            curve.append(accuracy_score(y_test, dummy))
            continue

        use = order_idx[:r]
        X_sub, y_sub = X_train[use], y_train[use]

        if len(np.unique(y_sub)) < 2:
            dummy = np.full_like(y_test, y_sub[0])
            curve.append(accuracy_score(y_test, dummy))
            continue

        model = LogisticRegression(max_iter=200, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sub, y_sub)
        curve.append(accuracy_score(y_test, model.predict(X_test)))

    return add_percentages, np.asarray(curve)


def random_baseline_addition(X_train, y_train, X_test, y_test, a, b, random_iters=10) -> Tuple[List[float], np.ndarray]:
    n = len(X_train)
    total_add = int(np.floor(b * n))
    step_add  = max(1, int(np.floor(a * n)))
    add_counts = list(range(0, total_add + 1, step_add))
    add_percentages = [100 * r / n for r in add_counts]

    curves = []
    for _ in range(random_iters):
        perm = np.random.permutation(n)
        curve = []
        for r in add_counts:
            if r == 0:
                maj = int(pd.Series(y_train).mode().iloc[0])
                dummy = np.full_like(y_test, maj)
                curve.append(accuracy_score(y_test, dummy))
                continue

            use = perm[:r]
            X_sub, y_sub = X_train[use], y_train[use]

            if len(np.unique(y_sub)) < 2:
                dummy = np.full_like(y_test, y_sub[0])
                curve.append(accuracy_score(y_test, dummy))
                continue

            model = LogisticRegression(max_iter=200, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            curve.append(accuracy_score(y_test, model.predict(X_test)))

        curves.append(curve)

    return add_percentages, np.mean(np.asarray(curves), axis=0)


# ----------------------------
# Noisy label detection
# ----------------------------
def flip_labels(y: np.ndarray, flip_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip flip_frac of labels. Assumes binary labels {0,1} or {-1,1} or general 2-class ints.
    Returns (y_flipped, flipped_mask_boolean).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    n = len(y)
    k = int(np.floor(flip_frac * n))
    idx = rng.choice(n, size=k, replace=False)

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got classes={classes}")

    # Flip by swapping to the other class
    c0, c1 = classes[0], classes[1]
    y[idx] = np.where(y[idx] == c0, c1, c0)

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return y, mask


def kmeans_noisy_detector(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    KMeans k=2 on 1D values; mark noisy if value < min(cluster_centers).
    Returns boolean mask of predicted noisy points.
    """
    v = np.asarray(values).reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    km.fit(v)
    centers = km.cluster_centers_.reshape(-1)
    thr = centers.min()
    return (v.reshape(-1) < thr)


# ----------------------------
# Main plotting per holdout base
# ----------------------------
def plot_six_lines_generic(x_pct: List[float],
                           curves: Dict[str, np.ndarray],
                           xlabel: str,
                           ylabel: str,
                           title: str,
                           save_path: str = None):
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
    })
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, layout="constrained")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for name, c in curves.items():
        ax.plot(x_pct, c * 100.0, lw=4, marker="o", label=name)

    ax.grid(True)
    ax.legend(loc="best")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_dir", type=str, required=True,
                        help="Directory containing {base}_Pred_{A|B|C|D} files.")
    parser.add_argument("--helper_path", type=str, default="PerformancePlots_4lines.py",
                        help="Path to PerformancePlots_4lines.py (contains PerformancePlot_Removal).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for PDFs. Default: <comparison_dir>/Plots")
    parser.add_argument("--a", type=float, default=0.05,
                        help="Step fraction (e.g., 0.05 means 5%% of training each step).")
    parser.add_argument("--b", type=float, default=0.40,
                        help="Total fraction to remove/add (e.g., 0.40 means up to 40%%).")
    parser.add_argument("--random_iters", type=int, default=10,
                        help="Random baseline iterations.")
    parser.add_argument("--flip_frac", type=float, default=0.10,
                        help="Fraction of labels to flip for noisy-label experiment.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--test_size", type=float, default=0.30,
                        help="Train/test split for plotting performance curves (proxy holdout).")
    args = parser.parse_args()

    comparison_dir = args.comparison_dir
    out_dir = args.out_dir or os.path.join(comparison_dir, "Plots")
    os.makedirs(out_dir, exist_ok=True)

    # Dynamic import of helper
    spec = importlib.util.spec_from_file_location("perfplots", args.helper_path)
    perfplots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perfplots)
    PerformancePlot_Removal = perfplots.PerformancePlot_Removal

    holdout_bases = [
        "characters_048_1",
        "Iris_1",
        "STL10_3568x_3",
        "Vehicle_01_1",
        "Volkert_15_2",
        "Volkert_1245_1",
    ]
    variants = ["A", "B", "C", "D"]

    # Collect noisy-label detection results
    noisy_rows = []

    for base in holdout_bases:
        # ---- load A/B/C/D dataframes ----
        dfs = {}
        for v in variants:
            fpath = find_file(comparison_dir, base, v)
            dfs[v] = load_table(fpath)

        # Basic schema checks
        for v in variants:
            if "Pred_Shap" not in dfs[v].columns:
                raise KeyError(f"[{base} Pred_{v}] missing column 'Pred_Shap'")
            if "target" not in dfs[v].columns:
                raise KeyError(f"[{base} Pred_{v}] missing column 'target'")

        # Use variant A as reference for shared columns like PerDShap/features
        df0 = dfs["A"].copy()
        if "PerDShap" not in df0.columns:
            raise KeyError(f"[{base}] missing column 'PerDShap' in Pred_A file (used as reference).")

        # Ensure row alignment across variants (best-effort)
        n0 = len(df0)
        for v in variants[1:]:
            if len(dfs[v]) != n0:
                raise ValueError(f"[{base}] row-count mismatch: Pred_A has {n0}, Pred_{v} has {len(dfs[v])}")

        # Feature columns inference
        exclude_cols = ["target", "Pred_Shap", "PerDShap"]
        feat_cols = infer_feature_columns(df0, exclude=exclude_cols)

        X = df0[feat_cols].to_numpy()
        y = df0["target"].to_numpy()

        # Proxy split: since files are "holdout test set" specific but do not include raw held-out features explicitly,
        # we use a deterministic train/test split here to produce consistent diagnostic curves.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

        # Build value arrays aligned to X_train indices
        # We must index the same rows used in X_train. train_test_split returns views by row order,
        # but not indices. So we re-create indices via a split on indices.
        idx = np.arange(len(df0))
        idx_train, idx_test = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, stratify=y
        )

        vals_map = {
            "Pred_A": dfs["A"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_B": dfs["B"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_C": dfs["C"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "Pred_D": dfs["D"].loc[idx_train, "Pred_Shap"].to_numpy(),
            "PerDShap": df0.loc[idx_train, "PerDShap"].to_numpy(),
        }

        # Subset train arrays to idx_train
        X_train2 = X[idx_train]
        y_train2 = y[idx_train]
        X_test2  = X[idx_test]
        y_test2  = y[idx_test]

        # --------------------------
        # Task 1: Removal plots (2)
        # --------------------------
        for order in ["descending", "ascending"]:
            save_path = os.path.join(out_dir, f"{base}_Removal_{order}.pdf")
            # helper automatically adds its own random baseline line named 'Random'
            _ = PerformancePlot_Removal(
                vals=vals_map,
                X_train=X_train2, y_train=y_train2,
                X_test=X_test2, y_test=y_test2,
                a=args.a, b=args.b,
                order=order,
                random_iters=args.random_iters,
                ylim=None,
                Names=list(vals_map.keys()),
                save_path=save_path
            )

        # --------------------------
        # Task 1: Addition plots (2)
        # --------------------------
        for order in ["descending", "ascending"]:
            curves = {}
            x_pct_ref = None

            # methods
            for name, v in vals_map.items():
                x_pct, c = performance_curve_addition(
                    vals=v, X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2,
                    a=args.a, b=args.b, order=order
                )
                curves[name] = c
                if x_pct_ref is None:
                    x_pct_ref = x_pct

            # random baseline
            x_pct_r, c_r = random_baseline_addition(
                X_train2, y_train2, X_test2, y_test2,
                a=args.a, b=args.b, random_iters=args.random_iters
            )
            curves["Random"] = c_r

            save_path = os.path.join(out_dir, f"{base}_Addition_{order}.pdf")
            plot_six_lines_generic(
                x_pct=x_pct_ref,
                curves=curves,
                xlabel="Fraction of data added (%)",
                ylabel="Accuracy (%)",
                title=f"{base} | Addition ({order})",
                save_path=save_path
            )

        # --------------------------------
        # Task 2: Noisy label detection (F1)
        # --------------------------------
        # Flip labels on *training portion* only (idx_train), and evaluate detection on that same set.
        y_train_clean = y_train2.copy()
        y_train_noisy, flipped_mask = flip_labels(y_train_clean, flip_frac=args.flip_frac, seed=args.seed)

        # run detector per method on values (1D) on training subset
        for method_name, v in vals_map.items():
            pred_noisy = kmeans_noisy_detector(v, seed=args.seed)
            f1 = f1_score(flipped_mask.astype(int), pred_noisy.astype(int), zero_division=0)
            noisy_rows.append({
                "holdout_base": base,
                "method": method_name,
                "flip_frac": args.flip_frac,
                "f1": float(f1),
                "n_train": int(len(y_train_clean)),
                "n_flipped": int(flipped_mask.sum()),
                "n_pred_noisy": int(pred_noisy.sum()),
            })

    # Save noisy-label summary
    noisy_df = pd.DataFrame(noisy_rows)
    noisy_csv = os.path.join(out_dir, "NoisyLabelDetection_F1.csv")
    noisy_df.to_csv(noisy_csv, index=False)

    # Also print a compact summary (mean F1 across holdouts)
    summary = (noisy_df
               .groupby("method")["f1"]
               .agg(["mean", "std", "count"])
               .reset_index()
               .sort_values("mean", ascending=False))
    print("\n=== Noisy Label Detection (F1) Summary across holdouts ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-holdout F1 to: {noisy_csv}")
    print(f"Plots saved under: {out_dir}")


if __name__ == "__main__":
    main()