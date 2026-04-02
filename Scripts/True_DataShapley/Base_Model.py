'''
Choice of Base Model
Logistic Regression is one of the base model. We now add several other base models and test them on Iris.
The parameter settings here are kept the same as those used in Per_DShap_Multiclass_Any.py.
'''
''' 0. Import Packages '''
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


''' 1. Function to Build Selected Base Models '''
def build_base_model(model_name = 'logistic', n_classes = None, random_state = None, **kwargs):
    model_name = model_name.lower()

    if model_name in ['logistic', 'logistic_regression', 'lr']:
        default_kwargs = {
            'max_iter': 200,
            'multi_class': 'multinomial',
            'random_state': random_state
        }
        default_kwargs.update(kwargs)
        return LogisticRegression(**default_kwargs)

    if model_name in ['random_forest', 'rf']:
        default_kwargs = {
            'n_estimators': 100,
            'random_state': random_state
        }
        default_kwargs.update(kwargs)
        return RandomForestClassifier(**default_kwargs)

    if model_name in ['decision_tree', 'dt', 'tree']:
        default_kwargs = {
            'random_state': random_state
        }
        default_kwargs.update(kwargs)
        return DecisionTreeClassifier(**default_kwargs)

    if model_name in ['svm']:
        default_kwargs = {
            'kernel': 'rbf',
            'decision_function_shape': 'ovr',
            'random_state': random_state
        }
        default_kwargs.update(kwargs)
        return SVC(**default_kwargs)

    if model_name in ['mlp', 'mlp_1layer', '1layer_mlp']:
        if n_classes is None:
            raise ValueError('n_classes should be provided when model_name = "mlp"!')
        default_kwargs = {
            'hidden_layer_sizes': (64,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 100,
            'random_state': random_state
        }
        default_kwargs.update(kwargs)
        return MLPClassifier(**default_kwargs)

    raise ValueError('Unsupported model_name! Available choices are logistic, random_forest, decision_tree, svm, and mlp.')


''' 2. Iris Data '''
iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, name = 'target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.5, stratify = y, random_state = 23
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


''' 3. Selected Base Models on Iris '''
base_models = [
    ('logistic_regression', 'Logistic Regression', {}),
    ('random_forest', 'Random Forest', {}),
    ('decision_tree', 'Decision Tree', {}),
    ('svm', 'SVM', {}),
    ('mlp_1layer', '1-layer MLP', {})
]

results = []

for model_name, model_title, model_kwargs in base_models:
    model = build_base_model(model_name = model_name,
                             n_classes = len(y.unique()),
                             random_state = 23, **model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        'Model': model_title,
        'Accuracy': round(float(accuracy), 6)
    })


''' 4. Print out Accuracy Table '''
results_df = pd.DataFrame(results)
print(results_df.to_string(index = False))
