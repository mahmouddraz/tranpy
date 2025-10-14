# TranPy Examples



## Notebooks

### 1. Getting Started
`01_getting_started.ipynb` - Minimal example to get started quickly
- Load dataset
- Train model
- Evaluate
- Predict
- Use pretrained models

### 2. Datasets
`02_datasets.ipynb` - Working with power system datasets
- List available datasets
- Load in different formats
- Train/test splitting
- Data statistics

### 3. Training Models
`03_training_models.ipynb` - Train classifiers from scratch
- Decision Trees
- Random Forest
- SVM
- Neural Networks
- Model comparison
- Save/load models

### 4. Pretrained Models
`04_pretrained_models.ipynb` - Using pretrained models
- List available models
- Filter by grid system
- Load and evaluate
- Compare models

### 5. Advanced Usage
`05_advanced_usage.ipynb` - Advanced techniques
- Hyperparameter tuning
- Cross-validation
- Feature importance
- Ensemble methods
- Probability calibration
- Learning curves

## Quick Start

```python
from tranpy.datasets import load_newengland
from tranpy.models import DecisionTreeClassifier

# Load data
X_train, X_test, y_train, y_test = load_newengland(test_size=0.2, random_state=42)

# Train
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

# Evaluate
results = model.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.3f}")
```

## Running Notebooks

```bash
jupyter notebook examples/
```

## Requirements

- tranpy
- jupyter
- numpy
- pandas
- scikit-learn
- matplotlib (for visualization)
