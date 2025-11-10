import csv
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression


def utility_k(X, y, X_test, y_test, subset_size, output_path):
    N = len(X)
    baseline = 0.0
    print(f"Generating subsets of size {subset_size} from {N} samples...")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'accuracy'])
        for subset in combinations(range(N), subset_size):
            subset = list(subset)
            X_subset = X.iloc[list(subset)]
            y_subset = y.iloc[list(subset)]

            if len(np.unique(y_subset)) < 2:
                acc = baseline
            else:
                model = LogisticRegression(max_iter=100)
                model.fit(X_subset, y_subset)
                acc = model.score(X_test, y_test)

            writer.writerow((subset, acc))
    print(f"Subsets of size {subset_size} generated and evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", required = True, help = 'Path to training features csv file')
    parser.add_argument("--y", required = True, help = 'Path to training labels csv file')
    parser.add_argument("--X_test", required = True, help = 'Path to test features csv file')
    parser.add_argument("--y_test", required = True, help = 'Path to test labels csv file')
    parser.add_argument("--subset_size", type=int, required = True, help = 'Size of subsets to evaluate')
    parser.add_argument("--output_path", required = True, help = 'Path to save the output csv file')
    args = parser.parse_args()

    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y).squeeze()
    X_test = pd.read_csv(args.X_test)
    y_test = pd.read_csv(args.y_test).squeeze()

    utility_k(X, y, X_test, y_test, args.subset_size, args.output_path)