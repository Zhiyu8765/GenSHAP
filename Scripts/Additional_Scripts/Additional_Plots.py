''' 250619 Experiment '''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
import numpy as np

from Scripts.Model.GNN_DShap import PERF_PATH

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
import os
import re
import numpy as np
import pandas as pd
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
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

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
import os
import re
import numpy as np
import pandas as pd
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
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

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