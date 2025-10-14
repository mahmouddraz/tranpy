#!/usr/bin/env python3
"""
Train all 18 classifiers on both datasets and save as pretrained models.

This script trains:
- 18 classifiers on New England 39-bus system
- 18 classifiers on IEEE 9-bus system
Total: 36 pretrained models

Models are saved to: src/tranpy/data/models/pretrained/
"""

import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tranpy.datasets import load_newengland, load_ieee9bus
from tranpy.models import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
    DummyClassifier,
    ExtraTreesClassifier,
    GaussianNB,
    GaussianProcessClassifier,
    GradientBoostingClassifier,
    KNeighborsClassifier,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    MLPClassifier,
    QuadraticDiscriminantAnalysis,
    RandomForestClassifier,
    RidgeClassifier,
    SGDClassifier,
    SVMClassifier,  # Note: it's SVMClassifier not SVC
    _LGBM_AVAILABLE,
    _XGB_AVAILABLE,
)

# Try to import optional models
HAS_LGBM = _LGBM_AVAILABLE
HAS_XGB = _XGB_AVAILABLE

if HAS_LGBM:
    from tranpy.models import LGBMClassifier
else:
    print("Warning: LGBMClassifier not available (install lightgbm)")

if HAS_XGB:
    from tranpy.models import XGBClassifier
else:
    print("Warning: XGBClassifier not available (install xgboost)")


def get_model_configs():
    """Define all 18 model configurations."""
    configs = [
        ("AdaBoostClassifier", AdaBoostClassifier, {"n_estimators": 50}),
        ("DecisionTreeClassifier", DecisionTreeClassifier, {"max_depth": 10}),
        ("DummyClassifier", DummyClassifier, {"strategy": "most_frequent"}),
        ("ExtraTreesClassifier", ExtraTreesClassifier, {"n_estimators": 100}),
        ("GaussianNB", GaussianNB, {}),
        ("GaussianProcessClassifier", GaussianProcessClassifier, {}),
        ("GradientBoostingClassifier", GradientBoostingClassifier, {"n_estimators": 100}),
        ("KNeighborsClassifier", KNeighborsClassifier, {"n_neighbors": 5}),
        ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis, {}),
        ("LogisticRegression", LogisticRegression, {"max_iter": 1000}),
        ("MLPClassifier", MLPClassifier, {"hidden_layer_sizes": (100,), "max_iter": 200}),
        ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis, {}),
        ("RandomForestClassifier", RandomForestClassifier, {"n_estimators": 100}),
        ("RidgeClassifier", RidgeClassifier, {}),
        ("SGDClassifier", SGDClassifier, {"max_iter": 1000}),
        ("SVC", SVMClassifier, {"kernel": "rbf", "probability": True}),  # Saved as SVC for compatibility
    ]

    if HAS_LGBM:
        configs.append(("LGBMClassifier", LGBMClassifier, {"n_estimators": 100}))

    if HAS_XGB:
        configs.append(("XGBClassifier", XGBClassifier, {"n_estimators": 100}))

    return configs


def train_and_save_model(model_class, params, X_train, y_train, X_test, y_test, save_path, model_name, grid_name):
    """Train a model and save it."""
    print(f"\nTraining {model_name} on {grid_name}...")

    try:
        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate
        results = model.evaluate(X_test, y_test, verbose=False)
        accuracy = results['accuracy']

        # Save model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        file_size_kb = save_path.stat().st_size / 1024

        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Saved to: {save_path.name} ({file_size_kb:.1f} KB)")

        return True, accuracy

    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        return False, None


def main():
    """Main training function."""
    print("=" * 80)
    print("TRAINING ALL PRETRAINED MODELS")
    print("=" * 80)

    # Output directory
    output_dir = Path(__file__).parent.parent / "src" / "tranpy" / "data" / "models" / "pretrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    print("\nLoading New England 39-bus system...")
    X_train_ne, X_test_ne, y_train_ne, y_test_ne = load_newengland(
        test_size=0.2, random_state=42
    )
    print(f"  ✓ Train: {len(X_train_ne)} samples")
    print(f"  ✓ Test: {len(X_test_ne)} samples")

    print("\nLoading IEEE 9-bus system...")
    X_train_9b, X_test_9b, y_train_9b, y_test_9b = load_ieee9bus(
        test_size=0.2, random_state=42
    )
    print(f"  ✓ Train: {len(X_train_9b)} samples")
    print(f"  ✓ Test: {len(X_test_9b)} samples")

    # Get model configurations
    model_configs = get_model_configs()
    print(f"\n{len(model_configs)} model types to train on 2 grids = {len(model_configs) * 2} total models")

    # Training statistics
    results = []

    # Train on New England
    print("\n" + "=" * 80)
    print("TRAINING ON NEW ENGLAND 39-BUS SYSTEM")
    print("=" * 80)

    for model_name, model_class, params in model_configs:
        filename = f"NewEngland_{model_name}.pkl"
        save_path = output_dir / filename

        success, accuracy = train_and_save_model(
            model_class, params,
            X_train_ne, y_train_ne, X_test_ne, y_test_ne,
            save_path, model_name, "New England"
        )

        results.append({
            'grid': 'NewEngland',
            'model': model_name,
            'success': success,
            'accuracy': accuracy
        })

    # Train on 9-Bus
    print("\n" + "=" * 80)
    print("TRAINING ON IEEE 9-BUS SYSTEM")
    print("=" * 80)

    for model_name, model_class, params in model_configs:
        filename = f"NineBusSystem_{model_name}.pkl"
        save_path = output_dir / filename

        success, accuracy = train_and_save_model(
            model_class, params,
            X_train_9b, y_train_9b, X_test_9b, y_test_9b,
            save_path, model_name, "9-Bus"
        )

        results.append({
            'grid': 'NineBusSystem',
            'model': model_name,
            'success': success,
            'accuracy': accuracy
        })

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n✓ Successfully trained: {successful}/{len(results)} models")
    if failed > 0:
        print(f"✗ Failed: {failed} models")
        print("\nFailed models:")
        for r in results:
            if not r['success']:
                print(f"  - {r['grid']}/{r['model']}")

    # Best models per grid
    print("\nBest models by accuracy:")
    for grid in ['NewEngland', 'NineBusSystem']:
        grid_results = [r for r in results if r['grid'] == grid and r['accuracy'] is not None]
        if grid_results:
            best = max(grid_results, key=lambda x: x['accuracy'])
            print(f"  {grid:20} {best['model']:35} {best['accuracy']:.4f}")

    print(f"\n✅ All models saved to: {output_dir}")
    print("\nTo use these models:")
    print('  from tranpy.models import load_pretrained')
    print('  model = load_pretrained("dt_ne39", local=True)')


if __name__ == "__main__":
    main()
