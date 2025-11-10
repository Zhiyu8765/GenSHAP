import os
import subprocess
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
output_dir = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42"
os.makedirs(output_dir, exist_ok=True)


'Iris'
m = 20
a = 0
b = 1
seed = 7282
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name = 'target')
iris_df = pd.concat([X, y], axis=1)

class_a = iris_df[iris_df["target"] == a].sample(n=m, random_state=i)
class_b = iris_df[iris_df["target"] == b].sample(n=m, random_state=i)
class_ab = pd.concat([class_a, class_b], axis=0).reset_index(drop=True)
class_ab['target'] = class_ab['target'].map({a:0, b:1})

X_train, X_test, y_train, y_test = train_test_split(class_ab.drop(columns='target'), class_ab['target'], test_size=0.5, stratify=class_ab['target'], random_state=seed)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'),  index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'),  index=False)
N = len(X_train)



'Breast Cancer'
m = 20
seed = 801
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target, name='target')
bc_df = pd.concat([X, y], axis=1)
bc_0 = bc_df[bc_df.target == 0]
bc_1 = bc_df[bc_df.target == 1]

bc0 = bc_0.sample(n=m, random_state=seed)
bc1 = bc_1.sample(n=m, random_state=seed)
bc = pd.concat([bc0, bc1], axis=0).reset_index(drop=True)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    bc.drop(columns='target'), bc.target, test_size=0.5, stratify=bc.target, random_state=i)
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train_raw)
X_test_scaled = scaler1.transform(X_test_raw)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train_pca)
X_test = scaler2.transform(X_test_pca)
X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'),  index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'),  index=False)


'MNIST'
m        = 20
seed0    = 2
class_a, class_b = 1, 7
i = 2

import tensorflow as tf
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

train_mask = (y_train_full == class_a) | (y_train_full == class_b)
X_pool = X_train_full[train_mask]
y_pool = y_train_full[train_mask]
X_pool = X_pool.reshape(len(X_pool), -1)
X_pool = pd.DataFrame(X_pool)
y_pool = pd.Series(y_pool, name='target')

idx_a = y_pool[y_pool == class_a].sample(n=m, random_state=seed0).index
idx_b = y_pool[y_pool == class_b].sample(n=m, random_state=seed0).index
idx = idx_a.union(idx_b)
X_all = X_pool.loc[idx].reset_index(drop=True)
y_all = y_pool.loc[idx].reset_index(drop=True).map({class_a: 0, class_b: 1})

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, stratify=y_all, random_state=seed0)
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train_raw)
X_test_scaled = scaler1.transform(X_test_raw)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train_pca)
X_test = scaler2.transform(X_test_pca)
X_train = pd.DataFrame(X_train, columns=[f'PC{i + 1}' for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=[f'PC{i + 1}' for i in range(X_test.shape[1])])

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'),  index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'),  index=False)
df_tr = X_train.copy()

# df_tr['target'] = y_train.values
# df_tr['TrueDShap'] = D_Shap
# df_tr.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_{i}", index=False)

save_dir = (
    "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/"
    "Data Valuation/DataShapley_Graph/"
    "Datasets/TrainnTestSets_V6/Group42"
)
shapley_model = DShap(
    X_train.values, y_train.values, X_test.values, y_test.values,
    num_test=m, sources=None, sample_weight=None,
    model_family='logistic', metric='accuracy', overwrite=True,
    directory='./temp', seed=250625 + seed0, max_iter=100, tol=1e-5
)
shapley_model.run(save_every=100, err=0.1, g_run=True, loo_run=True)

df_te = X_test.copy()
df_te['target'] = y_test.values
df_te.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_test_{i}", index=False)

df_tmc = X_train.copy()
df_tmc['TMC_Shap'] = shapley_model.vals_tmc
df_tmc.to_csv(f"{save_dir}/MNIST_{class_a}{class_b}_TMC_{i}", index=False)



'Automatic Prompt'
# for k in range(1, N):
#     out = f'utility_{k}.csv'
#     subprocess.run(['python', 'utility_k.py',
#                     '--X', 'X_train.csv',
#                     '--y', 'y_train.csv',
#                     '--X_test', 'X_test.csv',
#                     '--y_test', 'y_test.csv',
#                     '--subset_size', str(k),
#                     '--output_path', out]
#                    , check = True)
#     print(f'wrote {out}')
# subprocess.run(['python', 'DShap_integrate.py',
#                 '--N', str(N),
#                 '--path_pattern', 'utility_*.csv',
#                 '--output_file', 'shapley_values.csv']
#                , check = True)


python ./Scripts/True_DataShapley/utility_k.py \
  --X        "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/X_train.csv" \
  --y        "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/y_train.csv" \
  --X_test   "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/X_test.csv" \
  --y_test   "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/y_test.csv" \
  --subset_size 1 \
  --output_path "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/utils/utility1.csv"

python ./Scripts/True_DataShapley/DShap_integrate.py \
    --N 20 \
    --y_test "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/y_test.csv" \
    --X_train "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/X_train.csv" \
    --y_train "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/y_train.csv" \
    --path_pattern "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/utils/utility*.csv" \
    --output_file "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Datasets/TrainnTestSets_V6/Group42/MNIST_17_2"
