# TranPy

Machine learning tool for power system transient stability analysis with a simple, sklearn-style API.

## Quick Start

```bash
pip install tranpy
```

```python
from tranpy.datasets import load_newengland
from tranpy.models import SVMClassifier

X_train, X_test, y_train, y_test = load_newengland(test_size=0.2, random_state=42)
model = SVMClassifier(kernel='rbf')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

results = model.evaluate(X_test, y_test, verbose=True)
```

### XAI 

```python
from tranpy.explainers import SHAPExplainer

# Create explainer
explainer = SHAPExplainer(model, X_train, X_test)

# Generate global explanations
shap_values = explainer.explain_global()

# Visualize
explainer.plot_summary(save_path='shap_summary.png')

# Important features
top_features = explainer.get_top_features(n_features=10)
print(top_features)
``

Check out the [examples/](examples/) directory for Jupyter notebooks:

- `01_load_dataset.ipynb`
- `02_train_models.ipynb`
- `03_explainability.ipynb`
- `04_generate_data.ipynb` - > Requires PowerFactory local installation


