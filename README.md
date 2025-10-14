# TranPy: Power System Transient Stability Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TranPy is a modern Python library for **power system transient stability analysis** using machine learning and explainable AI. It provides:

- 🔌 **sklearn-style API** for power system stability prediction
- 🤖 **Multiple ML/DL models** (SVM, MLP, Decision Trees, DNN, RNN, Ensemble)
- 🔍 **Explainability tools** (LIME, SHAP, DALEX) to understand model decisions
- 📊 **Pre-generated datasets** from IEEE test systems (9-bus, New England 39-bus)
- ⚡ **PowerFactory integration** for generating custom datasets (optional)

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With all features (neural networks + explainability)
pip install -e ".[all]"

# Or install specific features
pip install -e ".[neural]"        # TensorFlow for DNN/RNN
pip install -e ".[explainability]" # LIME, SHAP, DALEX
pip install -e ".[dev]"           # Development tools
```

### Basic Usage

```python
from tranpy.datasets import load_newengland
from tranpy.models import SVMClassifier

# Load dataset (sklearn-style)
X_train, X_test, y_train, y_test = load_newengland(test_size=0.2, random_state=42)

# Train model
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

# Get top important features
top_features = explainer.get_top_features(n_features=10)
print(top_features)
```

## Features

### 1. Datasets Module

Load power system stability datasets like scikit-learn datasets:

```python
from tranpy.datasets import load_newengland, load_ieee9bus

# Load as dataset object
dataset = load_newengland()
print(dataset)  # StabilityDataset(grid='NewEngland', n_samples=..., n_features=78)

# Load with train/test split
X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)

# Load as simple X, y arrays
X, y = load_newengland(return_X_y=True)

# Load as pandas DataFrame
dataset = load_newengland(as_frame=True)
```

**Available Datasets:**
- `load_newengland()`: New England 39-bus system (78 features: 39 voltage magnitudes + 39 phase angles)
- `load_ieee9bus()`: IEEE 9-bus system (18 features)

### 2. Models Module

All models follow sklearn's estimator API:

**Classical ML Models:**
```python
from tranpy.models import SVMClassifier, MLPClassifier, DecisionTreeClassifier

# Support Vector Machine
svm = SVMClassifier(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)

# Multi-Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
mlp.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
```

**Deep Learning Models** (requires TensorFlow):
```python
from tranpy.models import DNNClassifier, RNNClassifier

# Deep Neural Network
dnn = DNNClassifier(hidden_layers=[10], epochs=20)
dnn.fit(X_train, y_train)

# Recurrent Neural Network (LSTM)
rnn = RNNClassifier(lstm_units=[150, 150], epochs=20)
rnn.fit(X_train, y_train)
```

**Ensemble Models:**
```python
from tranpy.models import EnsembleClassifier

# Default ensemble (SVM + MLP + DecisionTree + DNN)
ensemble = EnsembleClassifier(voting='soft')
ensemble.fit(X_train, y_train)

# Custom ensemble
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

# Explain single instance
explanation = explainer.explain_instance(X_test[0], num_features=10)

# Global explanations
sp_obj = explainer.explain_global(num_exps_desired=5)
explainer.plot_global_explanations(save_path='lime_global.pdf')

# Get top features
top_features = explainer.get_top_features(n_features=10)
```

**SHAP (SHapley Additive exPlanations):**
```python
from tranpy.explainers import SHAPExplainer

explainer = SHAPExplainer(model, X_train, X_test)

# Compute SHAP values
shap_values = explainer.explain_global()

# Visualizations
explainer.plot_summary(save_path='shap_summary.png')
explainer.plot_waterfall(instance_idx=0, save_path='shap_waterfall.png')

# Feature importance
top_features = explainer.get_top_features(n_features=10)
```

**DALEX (Break Down & Surrogate):**
```python
from tranpy.explainers import BreakdownExplainer, SurrogateExplainer

# Break Down: instance-level explanations
breakdown = BreakdownExplainer(model, X_train, X_test, y_test)
breakdown.explain_instance(instance_idx=0, save_path='breakdown.svg')

# Surrogate: interpretable approximation
surrogate = SurrogateExplainer(model, X_train, X_test, y_test)
feature_importances = surrogate.explain_global()
```

### 4. Simulation Module (Optional)

Generate custom datasets using DIgSILENT PowerFactory:

```python
from tranpy.simulation import PowerSystemSimulator

# Note: Requires PowerFactory installation
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

## Project Structure

```
pscc2026/
├── src/tranpy/              # Main package
│   ├── datasets/           # Dataset loaders
│   ├── models/             # ML/DL models
│   ├── explainers/         # XAI methods
│   ├── simulation/         # PowerFactory interface (optional)
│   └── utils/              # Utilities
├── tranpy/                  # Legacy code (for reference)
├── examples/                # Jupyter notebook examples
├── tests/                   # Unit tests
├── docs/                    # Documentation & paper
└── pyproject.toml          # Package configuration
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

## Related Projects

- **PowerFactory**: DIgSILENT power system simulation software
- **scikit-learn**: Machine learning library (API inspiration)
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- **DALEX**: Descriptive mAchine Learning EXplanations
