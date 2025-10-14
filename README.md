# TranPy

Machine learning tool for power system transient stability analysis with a simple, sklearn-style API.

## Related

scikit-learn, LIME, SHAP, DALEX, DIgSILENT PowerFactory.

## Quick Start

### Install

```bash
pip install -e .
```

### Example

```python
from tranpy.datasets import load_newengland
from tranpy.models import SVMClassifier

#load data 
X_train, X_test, y_train, y_test = load_newengland(test_size=0.2, random_state=42)
model = SVMClassifier(kernel='rbf')
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get detailed evaluation
results = model.evaluate(X_test, y_test, verbose=True)
```

### Using Explainability

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
```

## Features

### 1. Datasets Module

Load power system stability datasets like scikit-learn datasets:

```python
from tranpy.datasets import load_newengland, load_ieee9bus

# Load dataset
dataset = load_newengland()
print(dataset)  # StabilityDataset(grid='NewEngland', n_samples=..., n_features=78)

# Train/test split
X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)

# Load as simple X, y arrays
X, y = load_newengland(return_X_y=True)

# Load as  DataFrame
dataset = load_newengland(as_frame=True)
```

** Datasets:**
- `load_newengland()`: New England 39-bus system (78 features: 39 voltage magnitudes + 39 phase angles)
- `load_ieee9bus()`: IEEE 9-bus system (18 features)

### 2. Models 

All models follow sklearn's estimator API:

**Classical ML Models:**
```python
from tranpy.models import SVMClassifier, MLPClassifier, DecisionTreeClassifier


svm = SVMClassifier(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)


mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
mlp.fit(X_train, y_train)


dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
```

**Deep Learning Models** (requires TensorFlow):
```python
from tranpy.models import DNNClassifier, RNNClassifier

dnn = DNNClassifier(hidden_layers=[10], epochs=20)
dnn.fit(X_train, y_train)


rnn = RNNClassifier(lstm_units=[150, 150], epochs=20)
rnn.fit(X_train, y_train)
```

**Ensemble Models:**
```python
from tranpy.models import EnsembleClassifier

# Default (SVM + MLP + DecisionTree + DNN)
ensemble = EnsembleClassifier(voting='soft')
ensemble.fit(X_train, y_train)

# Custom 
from tranpy.models import SVMClassifier, MLPClassifier
ensemble = EnsembleClassifier(
    estimators=[
        ('svm', SVMClassifier()),
        ('mlp', MLPClassifier())
    ],
    voting='soft',
    weights=[0.6, 0.4]
)
```

**Pretrained Models:**
```python
from tranpy.models import load_pretrained, list_pretrained_models

# List available models
models = list_pretrained_models()

# Load pretrained model
model = load_pretrained('svm_ne39')
predictions = model.predict(X_test)
```

### 3. Explainers Module

Understand why models make certain predictions:

**LIME (Local Interpretable Model-agnostic Explanations):**
```python
from tranpy.explainers import LIMEExplainer

explainer = LIMEExplainer(model, X_train, X_test)

# Explain local
explanation = explainer.explain_instance(X_test[0], num_features=10)

# Explain global
sp_obj = explainer.explain_global(num_exps_desired=5)
explainer.plot_global_explanations(save_path='lime_global.pdf')

# Feature relevance
top_features = explainer.get_top_features(n_features=10)
```

**SHAP (SHapley Additive exPlanations):**
```python
from tranpy.explainers import SHAPExplainer

explainer = SHAPExplainer(model, X_train, X_test)

# SHAP values
shap_values = explainer.explain_global()

# Visualize
explainer.plot_summary(save_path='shap_summary.png')
explainer.plot_waterfall(instance_idx=0, save_path='shap_waterfall.png')

# Feature importance
top_features = explainer.get_top_features(n_features=10)
```

**DALEX (Break Down & Surrogate):**
```python
from tranpy.explainers import BreakdownExplainer, SurrogateExplainer

# Break Down
breakdown = BreakdownExplainer(model, X_train, X_test, y_test)
breakdown.explain_instance(instance_idx=0, save_path='breakdown.svg')

# Surrogate
surrogate = SurrogateExplainer(model, X_train, X_test, y_test)
feature_importances = surrogate.explain_global()
```

### 4. Simulation Module (Optional)

Generate custom datasets using DIgSILENT PowerFactory:
# Note: Requires PowerFactory installation

```python
from tranpy.simulation import PowerSystemSimulator


simulator = PowerSystemSimulator(
    grid='NewEngland',
    simulation_time=10
)

dataset = simulator.run(
    num_events=1000,
    fault_clearing_time=[12],
    max_load_change=[60]
)
```

## Examples

Check out the [examples/](examples/) directory for Jupyter notebooks:

- `01_load_dataset.ipynb`: Loading and exploring datasets
- `02_train_models.ipynb`: Training different models
- `03_explainability.ipynb`: Using explainability methods
- `04_generate_data.ipynb`: Generating custom datasets with PowerFactory

## Development

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Build package
python -m build
```

## Citation

If you use TranPy in your research, please cite:

```bibtex
@inproceedings{tranpy2026,
  title={TranPy: A Python Library for Power System Transient Stability Analysis with Explainable AI},
  author={Draz, Mahmoud and others},
  booktitle={PSCC 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Requirements

- Python 3.8+
- NumPy, pandas, scikit-learn, matplotlib (always)
- TensorFlow 2.8+ (optional, for neural networks)
- LIME, SHAP, DALEX (optional, for explainability)
- DIgSILENT PowerFactory (optional, for data generation)

