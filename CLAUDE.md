# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install for development (includes pytest, black, isort)
pip install -e ".[dev]"

# Install with optional features
pip install -e ".[explainability]"  # SHAP, LIME, DALEX
pip install -e ".[neural]"          # TensorFlow for DNN/RNN
pip install -e ".[boosting]"        # XGBoost, LightGBM
pip install -e ".[all]"             # Everything

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_basic_functionality.py -v

# Run a single test
pytest tests/test_basic_functionality.py::TestModels::test_svm_classifier -v

# Format code (line-length: 100)
black .
isort .
```

## Architecture

TranPy is a power system transient stability analysis library with an sklearn-compatible API.

### Module Structure (`src/tranpy/`)

- **models/**: ML classifiers inheriting from `StabilityClassifier` base class
  - `classical.py`: 14+ classifiers (SVM, RandomForest, MLP, DecisionTree, etc.)
  - `neural.py`: TensorFlow-based DNNClassifier, RNNClassifier (requires `[neural]`)
  - `ensemble.py`: EnsembleClassifier for model combination
  - `pretrained.py`: Load/save pretrained models from registry

- **datasets/**: Dataset loaders returning sklearn-style data
  - `load_newengland()`, `load_ieee9bus()` - return train/test split or StabilityDataset
  - Bundled data in `data/datasets/*.pickle`

- **explainers/**: XAI methods (all optional dependencies)
  - `SHAPExplainer`, `LIMEExplainer`, `BreakdownExplainer`

- **simulation/**: DIgSILENT PowerFactory integration for generating training data
  - Requires local PowerFactory installation

### Key Patterns

1. **Optional dependencies**: Handled via try/except with availability flags (`_XGB_AVAILABLE`, `_NEURAL_AVAILABLE`, etc.)

2. **Model interface**: All classifiers have `fit()`, `predict()`, `score()`, `evaluate()` methods

3. **Dataset loading**:
   ```python
   # Returns train/test split
   X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)

   # Returns StabilityDataset object
   dataset = load_newengland()
   ```

## Important Notes

- Package name on PyPI is `tranpy-ai`, but import as `tranpy`
- Python 3.8-3.12 supported
- CI tests run on all Python versions via GitHub Actions
