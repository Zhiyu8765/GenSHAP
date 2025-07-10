import os
import subprocess
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
output_dir = "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02"
os.makedirs(output_dir, exist_ok=True)

m = 30
seed = 708
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name = 'target')
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
iris_df = pd.concat([X_scaled, y], axis=1)

iris_1 = iris_df[iris_df['target'] == 1].sample(m, random_state=seed)
iris_2 = iris_df[iris_df['target'] == 2].sample(m, random_state=seed)
iris_12 = pd.concat([iris_1, iris_2], axis = 0).copy()
iris_12['target'] = iris_12['target'].map({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(iris_12.drop(columns='target'), iris_12['target'], test_size=0.5, random_state=seed)
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test .to_csv(os.path.join(output_dir, 'X_test.csv'),  index=False)
y_test .to_csv(os.path.join(output_dir, 'y_test.csv'),  index=False)
N = len(X_train)

for k in range(1, N):
    out = f'utility_{k}.csv'
    subprocess.run(['python', 'utility_k.py',
                    '--X', 'X_train.csv',
                    '--y', 'y_train.csv',
                    '--X_test', 'X_test.csv',
                    '--y_test', 'y_test.csv',
                    '--subset_size', str(k),
                    '--output_path', out]
                   , check = True)
    print(f'wrote {out}')
subprocess.run(['python', 'DShap_integrate.py',
                '--N', str(N),
                '--path_pattern', 'utility_*.csv',
                '--output_file', 'shapley_values.csv']
               , check = True)


python ./Scripts/True_DataShapley/utility_k.py \
  --X        "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02/X_train.csv" \
  --y        "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02/y_train.csv" \
  --X_test   "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02/X_test.csv" \
  --y_test   "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02/y_test.csv" \
  --subset_size 19 \
  --output_path "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial02/utils/utility19.csv"

python ./Scripts/True_DataShapley/DShap_integrate.py \
    --N 20 \
    --path_pattern "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial01/utils/utility_*.csv" \
    --output_file "/Users/cuhksz-ml-01/Dropbox/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250708/Trial01/DShap_20.csv"