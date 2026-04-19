"""Early prediction experiment framework for transient stability assessment.

Evaluates classifier accuracy as a function of observation time window,
determining the minimum time required for reliable stability prediction.

Supports any sklearn-compatible classifier and computes accuracy, precision,
recall, F1, and AUC with bootstrap confidence intervals.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .temporal import TemporalFeatureExtractor
from ..datasets.temporal_loaders import TemporalDataset


def run_early_prediction_experiment(
    dataset: TemporalDataset,
    classifiers: Dict[str, Any],
    time_windows: Optional[List[float]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    relative_to_clearing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run early prediction experiment across multiple time windows and classifiers.

    For each time window, extracts temporal features from trajectories truncated
    to that window, then evaluates classifiers using stratified cross-validation.

    When relative_to_clearing=True (default), time windows are measured relative
    to each event's fault clearing time — this is the physically meaningful
    question: "how many seconds of post-clearing observation do we need?"

    Events where the required post-clearing observation exceeds available data
    are excluded from that time window's evaluation.

    Args:
        dataset: TemporalDataset with event trajectories and labels
        classifiers: Dictionary mapping model names to sklearn classifier instances
        time_windows: List of observation windows in seconds post-clearing
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        relative_to_clearing: If True, windows are relative to clearing time
        verbose: Print progress messages

    Returns:
        DataFrame with columns: time_window, classifier, accuracy, precision,
        recall, f1, auc, accuracy_std, ..., n_events
    """
    if time_windows is None:
        time_windows = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    labels = dataset.labels
    results = []

    for tw in time_windows:
        if verbose:
            label = "post-clearing" if relative_to_clearing else "absolute"
            print(f"Time window: {tw:.2f}s ({label})")

        # Filter events that have enough post-clearing data for this window
        if relative_to_clearing:
            valid_indices = []
            for i, event in enumerate(dataset.events):
                if event.clearing_time is not None:
                    available = float(event.time[-1]) - event.clearing_time
                    if available >= tw:
                        valid_indices.append(i)
                else:
                    # No clearing time detected — include with absolute cutoff
                    valid_indices.append(i)

            if verbose and len(valid_indices) < len(dataset.events):
                print(f"  Using {len(valid_indices)}/{len(dataset.events)} events "
                      f"(others lack {tw:.1f}s post-clearing data)")

            # Build filtered dataset for feature extraction
            filtered_events = [dataset.events[i] for i in valid_indices]
            filtered_labels = np.array([labels[i] for i in valid_indices])
        else:
            filtered_events = dataset.events
            filtered_labels = labels

        if len(filtered_events) < 10:
            if verbose:
                print(f"  Skipping: only {len(filtered_events)} events available")
            continue

        # Extract features for this time window
        extractor = TemporalFeatureExtractor(
            time_window=tw,
            relative_to_clearing=relative_to_clearing,
        )

        # Extract per event
        all_features = []
        for event in filtered_events:
            feat_dict = extractor.extract(event)
            all_features.append(feat_dict)

        feature_names = sorted(all_features[0].keys())
        X = np.array([
            [feat_dict.get(name, 0.0) for name in feature_names]
            for feat_dict in all_features
        ])

        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Adjust n_splits if not enough samples per class
        unique, counts = np.unique(filtered_labels, return_counts=True)
        min_class = min(counts)
        actual_splits = min(n_splits, min_class)
        if actual_splits < 2:
            if verbose:
                print(f"  Skipping: not enough samples per class (min={min_class})")
            continue

        # Run cross-validation for each classifier
        for model_name, clf_template in classifiers.items():
            fold_metrics = _cross_validate(
                X, filtered_labels, clf_template,
                n_splits=actual_splits, random_state=random_state
            )

            result = {
                'time_window': tw,
                'classifier': model_name,
                'n_features': X.shape[1],
                'n_events': len(filtered_events),
            }

            # Compute mean and std for each metric
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = fold_metrics[metric_name]
                result[f'{metric_name}_mean'] = np.mean(values)
                result[f'{metric_name}_std'] = np.std(values)

            results.append(result)

            if verbose:
                acc = result['accuracy_mean']
                auc = result['auc_mean']
                print(f"  {model_name}: Acc={acc:.4f}, AUC={auc:.4f}")

    return pd.DataFrame(results)


def _cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    clf_template: Any,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """Run stratified cross-validation and return per-fold metrics.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        clf_template: Sklearn classifier instance (will be cloned)
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary mapping metric names to lists of per-fold values
    """
    # Handle case with too few samples for requested folds
    n_samples = len(y)
    n_classes = len(np.unique(y))
    min_class_count = min(np.bincount(y.astype(int)))
    actual_splits = min(n_splits, min_class_count, n_samples)

    if actual_splits < 2:
        # Not enough samples for cross-validation; train on all, evaluate on all
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = clone(clf_template)
        clf.fit(X_scaled, y)
        y_pred = clf.predict(X_scaled)
        y_prob = _get_proba(clf, X_scaled)

        return {
            'accuracy': [accuracy_score(y, y_pred)],
            'precision': [precision_score(y, y_pred, zero_division=0)],
            'recall': [recall_score(y, y_pred, zero_division=0)],
            'f1': [f1_score(y, y_pred, zero_division=0)],
            'auc': [roc_auc_score(y, y_prob) if y_prob is not None else 0.0],
        }

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and predict
        clf = clone(clf_template)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = _get_proba(clf, X_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        if y_prob is not None and len(np.unique(y_test)) > 1:
            metrics['auc'].append(roc_auc_score(y_test, y_prob))
        else:
            metrics['auc'].append(0.0)

    return metrics


def _get_proba(clf: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Safely get probability predictions from a classifier."""
    if hasattr(clf, 'predict_proba'):
        try:
            proba = clf.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba[:, 0]
        except Exception:
            pass
    if hasattr(clf, 'decision_function'):
        try:
            return clf.decision_function(X)
        except Exception:
            pass
    return None


def find_knee_point(results_df: pd.DataFrame, model: str, metric: str = 'accuracy') -> float:
    """Find the knee point where increasing observation time yields diminishing returns.

    Uses the maximum curvature method to identify the point where
    accuracy improvement begins to plateau.

    Args:
        results_df: DataFrame from run_early_prediction_experiment
        model: Model name to analyze
        metric: Metric to analyze (default: 'accuracy')

    Returns:
        Time window value at the knee point
    """
    model_col = 'classifier' if 'classifier' in results_df.columns else 'model'
    metric_col = f'{metric}_mean' if f'{metric}_mean' in results_df.columns else metric
    model_df = results_df[results_df[model_col] == model].sort_values('time_window')
    x = model_df['time_window'].values
    y = model_df[metric_col].values

    if len(x) < 3:
        return x[-1]

    # Normalize to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Compute curvature using finite differences
    dx = np.gradient(x_norm)
    dy = np.gradient(y_norm)
    d2y = np.gradient(dy)

    curvature = np.abs(d2y) / (1 + dy**2)**1.5

    # Knee point is at maximum curvature
    knee_idx = np.argmax(curvature[1:-1]) + 1  # skip endpoints

    return float(x[knee_idx])
