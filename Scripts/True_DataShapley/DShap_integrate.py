import pandas as pd
import numpy as np
import argparse
import math
from itertools import combinations
from ast import literal_eval
from glob import glob

def load_utilities(path_pattern):
    files = sorted(glob(path_pattern))
    util_dict = {}
    for file in files:
        df = pd.read_csv(file)
        df["subset"] = df["subset"].apply(literal_eval)
        for subset, utility in zip(df["subset"], df["accuracy"]):
            key = tuple(sorted(subset))
            util_dict[key] = utility
    return util_dict

def compute_shapley(N, util_dict, baseline):
    shap = np.zeros(N)
    all_indices = list(range(N))
    for i in all_indices:
        others = [j for j in range(N) if j != i]
        for s in range(0, len(others) + 1):
            weight = 1/(N * math.comb(N-1, s))
            for S in combinations(others, s):
                key_S = tuple(sorted(S))
                key_Si = tuple(sorted(S + (i,)))
                v_S = util_dict.get(key_S, baseline)
                v_Si = util_dict.get(key_Si, baseline)
                shap[i] += weight * (v_Si - v_S)
    return shap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True,
                        help="Number of data points")
    parser.add_argument("--path_pattern", type=str, required=True,
                        help="Path pattern to load utility files")
    parser.add_argument("--y_test", type=str, required=True,
                        help="Path to test labels CSV for baseline computation")  # CHANGED: new arg
    parser.add_argument("--X_train", type=str, required=True,
                        help="Path to training features CSV")  # CHANGED: new arg
    parser.add_argument("--y_train", type=str, required=True,
                        help="Path to training labels CSV")  # CHANGED: new arg
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file to save merged training data with Shapley values")  # CHANGED: updated help
    args = parser.parse_args()
    util_dict = load_utilities(args.path_pattern)
    y_test = pd.read_csv(args.y_test).squeeze()
    baseline = max(np.mean(y_test == c) for c in np.unique(y_test))
    shapley_vals = compute_shapley(args.N, util_dict, baseline)
    X_train = pd.read_csv(args.X_train)
    y_train = pd.read_csv(args.y_train).squeeze()
    df_shap = pd.DataFrame({"shapley_value": shapley_vals})
    df_out = pd.concat([X_train.reset_index(drop=True),
                        y_train.reset_index(drop=True),
                        df_shap.reset_index(drop=True)], axis=1)
    df_out.to_csv(args.output_file, index=False)
    print(f"Shapley values with training data saved to {args.output_file}")