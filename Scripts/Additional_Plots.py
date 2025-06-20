import matplotlib.pyplot as plt
import pandas as pd

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