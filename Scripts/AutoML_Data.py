import numpy as np
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess
iris = load_iris()
X, y = iris['data'], iris['target']
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

n_train = len(X_train)
data_weights = np.ones(n_train)

# Sampling function based on weights
def get_weighted_sample(X, y, weights, sample_size=None):
    probs = weights / np.sum(weights)
    indices = np.random.choice(np.arange(len(X)), size=sample_size or len(X), p=probs, replace=True)
    return X[indices], y[indices]

# Gradient-free optimization
def train_weights_with_tpot(X_train, y_train, X_val, y_val, data_weights, n_epochs=30, lr=0.1):
    for epoch in range(n_epochs):
        X_sampled, y_sampled = get_weighted_sample(X_train, y_train, data_weights)

        try:
            tpot = TPOTClassifier(generations=5, population_size=20, verbosity=0,
                                  max_time_mins=1, config_dict='TPOT light',
                                  disable_update_check=True)
            tpot.fit(X_sampled, y_sampled)
            preds = tpot.predict(X_val)
            val_acc = accuracy_score(y_val, preds)
        except Exception as e:
            print("TPOT failed:", e)
            val_acc = 0.0

        # Reward high-weighted points if accuracy was good
        probs = data_weights / np.sum(data_weights)
        reward = val_acc
        data_weights += lr * reward * probs  # simple REINFORCE-style update
        data_weights = np.clip(data_weights, 1e-4, 10.0)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Val Accuracy = {val_acc:.4f}")

    # Normalize and return
    return data_weights / np.max(data_weights)

# Run it
final_weights = train_weights_with_tpot(X_train, y_train, X_val, y_val, data_weights)

# --- 5. Reuse your performance plot function as-is ---
def PerformancePlot_Removal(vals, X_train, y_train, X_test, y_test, a, b, order='descending', random_iters=10, save_path=None):
    if not isinstance(vals, (list, tuple)):
        vals = [vals]
    n = len(X_train)
    total_remove_count = int(np.floor(b * n))
    step_remove_count = max(1, int(np.floor(a * n)))
    removal_counts = list(range(0, total_remove_count + 1, step_remove_count))
    removal_percentages = [(count / n) * 100 for count in removal_counts]
    results = {}

    def compute_curve(ordering):
        curve = []
        for r_count in removal_counts:
            keep_indices = ordering[r_count:]
            if len(keep_indices) == 0:
                curve.append(0.0)
                continue
            X_sub = X_train[keep_indices]
            y_sub = y_train[keep_indices]
            unique_classes = np.unique(y_sub)
            if len(unique_classes) < 2:
                dummy_pred = np.full_like(y_test, fill_value=unique_classes[0])
                acc = accuracy_score(y_test, dummy_pred)
                curve.append(acc)
                continue
            model = LogisticRegression(max_iter=5000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            curve.append(acc)
        return np.array(curve)

    for i, val in enumerate(vals):
        if order == 'ascending':
            sorted_indices = np.argsort(val)
        else:
            sorted_indices = np.argsort(val)[::-1]
        results[f'valuation {i + 1}'] = compute_curve(sorted_indices)

    random_curves = []
    for _ in range(random_iters):
        random_order = np.random.permutation(n)
        random_curves.append(compute_curve(random_order))
    random_curve = np.mean(random_curves, axis=0)
    results['Random Baseline'] = random_curve

    plt.figure(figsize=(8, 8))
    plt.xlabel('Percentage of training data removed (%)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    for key, curve in results.items():
        plt.plot(removal_percentages, curve * 100, lw=3, marker='o', label=key)
    plt.legend(fontsize=14)
    plt.title('Performance vs. Data Removal', fontsize=18)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return removal_percentages, results

# --- 6. Plot it ---
PerformancePlot_Removal(
    vals=final_weights,
    X_train=X_train,
    y_train=y_train,
    X_test=X_val,
    y_test=y_val,
    a=0.1,
    b=0.5,
    order='ascending',
    random_iters=5,
    save_path='/Users/cuhksz-ml-01/Nutstore Files/CUHK-SZ/Research/Data Valuation/DataShapley_Graph/Outputs/250324/Removal_Iris_Auto_Low.png'
)