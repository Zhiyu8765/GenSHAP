'''
0. Import Packages
'''
import time
import json
import os
import warnings
import numpy as np
import pandas as pd
import inspect
from math import ceil, sqrt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from numpy.typing import ArrayLike
from collections.abc import Mapping as MappingABC
from typing import Callable, Dict, Mapping, List, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pydvl.valuation.types import Sample
from pydvl.valuation.methods.banzhaf import BanzhafValuation
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation import (Dataset, SupervisedScorer, ModelUtility, ShapleyValuation,
                             BetaShapleyValuation, DeltaShapleyValuation, GroupTestingShapleyValuation, ClasswiseShapleyValuation)
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility
from pydvl.valuation.samplers.classwise import ClasswiseSampler
from pydvl.valuation.samplers import PermutationSampler, OwenSampler, GridOwenStrategy
from pydvl.valuation.samplers.powerset import UniformSampler, NoIndexIteration
from pydvl.valuation.samplers.truncation import RelativeTruncation
from pydvl.valuation.stopping import (RankCorrelation, MinUpdates, MaxSamples, History, NoStopping)
from pydvl.valuation.methods.knn_shapley import KNNShapleyValuation
from pydvl.valuation.samplers.stratified import DeltaShapleyNCSGDSampleSize
from pydvl.valuation.samplers.stratified import StratifiedSampler, PowerLawSampleSize


' For Calculation of TMC Data Shapley Values'
### Check Working Directory
import sys
print(sys.executable)
import os
# os.chdir('/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/DataShapley')
os.chdir('/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/DataShapley')
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

#%matplotlib inline
MEM_DIR = '../'


'''
1. Load Data
'''
TRAIN_CSV = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Volkert_Multi/volkert_1245_1.csv"
TEST_CSV  = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Volkert_Multi/volkert_1245_test_1.csv"
TMC_CSV   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Volkert_Multi/volkert_1245_TMC_1.csv"
OUT_DIR   = "/mnt/data/ZhiyuChen/pycharm_project_236/tmp/pycharm_project_236/Datasets/1000pts_I/Multi_V1/Baselines/Volkert_Multi"
LABEL_COL = "target"
runtimes = {}
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
drop_cols = [LABEL_COL, "PerDShap"]
drop_cols = [c for c in drop_cols if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).to_numpy()
y_train = df_train[LABEL_COL].to_numpy()
X_test  = df_test.drop(columns=[LABEL_COL]).to_numpy()
y_test  = df_test[LABEL_COL].to_numpy()
train_ds = Dataset(X_train, y_train)
test_ds  = Dataset(X_test,  y_test)
N = len(train_ds)

def timed_fit(name, val_obj, train_ds):
    t0 = time.perf_counter()
    val_obj.fit(train_ds)
    dt = time.perf_counter() - t0
    runtimes[name] = dt
    print(f"[runtime] {name}: {dt:.3f} s")
    return val_obj

# ---------- Base models ----------
# (A) Logistic regression for model-based methods
lr = LogisticRegression(max_iter=200)
scorer = SupervisedScorer("accuracy", test_ds, default=0.0, range=(0.0, 1.0))
utility = ModelUtility(lr, scorer)

# ---------- Common stoppers ----------
TMC_MAX_ITER = 3 * N
MC_BUDGET = 3 * N
BURN_IN = max(200, int(0.4 * MC_BUDGET))
RANK_RTOL = 1e-3
shapley_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
delta_stopping   = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
group_stopping   = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
banzhaf_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)
semi_stopping    = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)

# Parameter settings in Beta Shapley
BETA_ALPHA, BETA_BETA = 1, 16
beta_stopping = MinUpdates(MC_BUDGET) | RankCorrelation(rtol=RANK_RTOL, burn_in=BURN_IN)

# Parameter settings in Owen Shapley
OWEN_TARGET_M = max(2000, int(38.4 * N))
N_SAMPLES_INNER = 2
N_SAMPLES_OUTER = int(np.ceil(OWEN_TARGET_M / N_SAMPLES_INNER))

# Parameter settings in KNN Shapley
KNN_K = min(10, max(1, N - 1))

'''
2. Compute Valuations
'''
''' A. Approximation of Data Shapley '''
' 1. TMC Shapley'
model = 'logistic'
problem = 'classification'
metric = 'accuracy'
directory = './temp'
shapley_model = DShap(X_train, y_train,
                      X_test, y_test,
                      num_test = len(X_test), sources=None,
                      sample_weight=None, model_family=model,
                      metric=metric, overwrite=True,
                      directory=directory, seed=251013,
                      max_iter=TMC_MAX_ITER, tol=0.05)
t0 = time.perf_counter()
shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)
runtimes["TMC Shapley"] = time.perf_counter() - t0
print(f"[runtime] TMC Shapley: {runtimes['TMC Shapley']:.3f} s")
tmc_vals = np.asarray(shapley_model.vals_tmc).reshape(-1)
Vehicle_TMC = pd.DataFrame(X_train)  # columns become 0..d-1
Vehicle_TMC["TMC_Shap"] = tmc_vals


'2. Delta Shapley'
l, u = N // 3, (2 * N) // 3
size_strategy = PowerLawSampleSize(lower_bound=l, upper_bound=u, exponent=0.0, n_samples = MC_BUDGET)
sampler = StratifiedSampler(sample_sizes=size_strategy, seed=42)
val_delta = DeltaShapleyValuation(utility=utility, sampler=sampler, is_done=delta_stopping, progress=True)
timed_fit("DeltaShapley", val_delta, train_ds)
delta_vals = val_delta.result.values


'3. Group-Testing Shapley'
val_group = GroupTestingShapleyValuation(utility=utility, progress=True, n_samples = MC_BUDGET, epsilon=0.05)
timed_fit("GroupTesting", val_group, train_ds)
group_vals = val_group.result.values


'4. Owen Shapley'
owen_sampler = OwenSampler(outer_sampling_strategy=GridOwenStrategy(n_samples_outer=N_SAMPLES_OUTER), n_samples_inner=N_SAMPLES_INNER)
owen_stop = NoStopping(owen_sampler)
owen_val = ShapleyValuation(utility, owen_sampler, owen_stop, progress=True)

from joblib import parallel_config
t0 = time.perf_counter()
with parallel_config(n_jobs=4):
    owen_val.fit(train_ds)
runtimes["OwenShapley"] = time.perf_counter() - t0
print(f"[runtime] OwenShapley: {runtimes['OwenShapley']:.3f} s")
res_owen = owen_val.result.values



''' B. Generalisation of Data Shapley '''
'5. Beta Shapley'
# Use recommended settings from original paper: alpha=1, beta=16
val_beta = BetaShapleyValuation(
    utility=utility,
    sampler=PermutationSampler(),
    is_done=beta_stopping,
    alpha=BETA_ALPHA,
    beta=BETA_BETA,
    progress=True
)
timed_fit("BetaShapley", val_beta, train_ds)
beta_vals = val_beta.result.values


'6. Data Banzhaf'
val_banzhaf = BanzhafValuation(
    utility=utility,
    sampler=PermutationSampler(),
    is_done=banzhaf_stopping,
    progress=True
)
timed_fit("DataBanzhaf", val_banzhaf, train_ds)
banzhaf_vals = val_banzhaf.result.values


'7. Semi-Values'
class UniformSemiValue(SemivalueValuation):
    def log_coefficient(self, subset_size: int, total_size: int) -> float:
        return 0.0  # log(1.0) for uniform weight
val_semiv = UniformSemiValue(
    utility=utility,
    sampler=PermutationSampler(),
    is_done=semi_stopping,
    progress=True
)
timed_fit("UniformSemiValue", val_semiv, train_ds)
semivalue_vals = val_semiv.result.values



''' C. Special Case of Data Shapley '''
'8. KNN Shapley'
from pydvl.valuation.methods.knn_shapley import KNNShapleyValuation
# Practical default based on baseline literature (K=10)
knn_model = KNeighborsClassifier(n_neighbors=KNN_K)
val_knn = KNNShapleyValuation(model=knn_model, test_data=test_ds, progress=True)
from joblib import parallel_config
t0 = time.perf_counter()
with parallel_config(n_jobs=4):
    val_knn.fit(train_ds)
runtimes["KNNShapley"] = time.perf_counter() - t0
print(f"[runtime] KNNShapley: {runtimes['KNNShapley']:.3f} s")
knn_vals = val_knn.result.values

'9. Leave-one-out'
clf_full = clone(lr)
clf_full.fit(X_train, y_train)
full_score = accuracy_score(y_test, clf_full.predict(X_test))
loo_vals = np.empty(len(X_train), dtype=float)
for i in range(len(X_train)):
    mask = np.ones(len(X_train), dtype=bool)
    mask[i] = False
    clf_sub = clone(lr)
    clf_sub.fit(X_train[mask], y_train[mask])
    score_i = accuracy_score(y_test, clf_sub.predict(X_test))
    loo_vals[i] = full_score - score_i



''' D. Influence Functions '''
'10. CG Influence'
import torch
from torch.utils.data import TensorDataset, DataLoader
from pydvl.influence.torch import CgInfluence
class TorchLogistic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

# Prepare Tensors & DataLoader
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).long()

# Instantiate model and loss
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
torch_model = TorchLogistic(input_dim, num_classes)
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
for _ in range(100):
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = torch_model(bx)
        loss = loss_fn(out, by)
        loss.backward()
        optimizer.step()
infl_cg = CgInfluence(
    model=torch_model,
    loss=loss_fn,
    regularization=1e-3,
    rtol=1e-5,
    atol=1e-6,
    maxiter=None,
    progress=True
)
# Fit influence model
infl_cg.fit(train_loader)
# Compute influence of training points on test set
# result shape: (n_train, n_test)
infl_matrix = infl_cg.influences(X_test_t, y_test_t, X_train_t, y_train_t)
cg_influence_vals = infl_matrix.sum(axis=1).detach().numpy()


''' Collect and Save'''
shapley_df = pd.DataFrame({
    "TMC Shapley": tmc_vals,
    "Delta Shapley": delta_vals,
    "Group-Testing Shapley": group_vals,
    "Owen Shapley": res_owen,
    "Beta Shapley": beta_vals,
    "Data Banzhaf": banzhaf_vals,
    "Uniform SemiValue": semivalue_vals,
    "KNN Shapley": knn_vals,
    "LOO": loo_vals,
    "CG Influence": cg_influence_vals
})


# Save runtime
pd.DataFrame([runtimes]).to_csv(os.path.join(OUT_DIR, "volkert_1245_1_runtimes.csv"), index=False)

# Save combined predictions and valuations
df_train = df_train.reset_index(drop=True)
combined_df = pd.concat([df_train, shapley_df], axis=1)
combined_df.to_csv(os.path.join(OUT_DIR, "volkert_1245_1_all_baselines.csv"), index=False)