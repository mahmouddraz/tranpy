#!/usr/bin/env python3
"""Phase 6: Deep Learning Models for Temporal Stability Prediction.

Evaluates LSTM, TCN, and Transformer models on temporal features
extracted from NineBusSystem and NewEngland datasets (n=500 events).

Compares deep learning models against best classical ML baselines
from Phase 5 (ETrees, RF) across multiple time windows.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tranpy.datasets.temporal_loaders import load_temporal_dataset
from tranpy.features.temporal import TemporalFeatureExtractor
from tranpy.models.neural import LSTMClassifier, TCNClassifier, TransformerClassifier

N_EVENTS = 500
TIME_WINDOWS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'papers', 'ieeaccess25')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'Images', 'temporal')


def get_dl_models():
    """Get deep learning model instances."""
    return {
        'LSTM': LSTMClassifier(
            lstm_units=[64, 32], dropout=0.2, use_attention=True,
            dense_units=32, learning_rate=1e-3, epochs=100, batch_size=32,
        ),
        'TCN': TCNClassifier(
            n_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8, 16],
            dropout=0.2, dense_units=32, learning_rate=1e-3, epochs=100, batch_size=32,
        ),
        'Transformer': TransformerClassifier(
            d_model=64, n_heads=4, n_layers=2, ff_dim=128,
            dropout=0.1, dense_units=32, learning_rate=1e-3, epochs=100, batch_size=32,
        ),
    }


def get_baseline_models():
    """Get classical ML baselines from Phase 5."""
    return {
        'ETrees': ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }


def cv_evaluate(X, y, model_name, model_factory, n_splits=5):
    """Run stratified cross-validation for a model.

    For DL models, creates a fresh instance per fold. For sklearn models, uses clone.

    Returns dict with mean/std of accuracy, precision, recall, f1, auc.
    """
    min_class = min(np.bincount(y.astype(int)))
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        actual_splits = 2

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        # DL models: create fresh instance each fold
        if isinstance(model_factory, dict):
            # model_factory is a dict of params for the model class
            model = model_factory['class'](**model_factory['params'])
        elif hasattr(model_factory, 'fit'):
            # It's a model instance — clone for sklearn, recreate for DL
            if model_name in ('LSTM', 'TCN', 'Transformer'):
                model = model_factory.__class__(**{
                    k: v for k, v in model_factory.__dict__.items()
                    if not k.endswith('_')
                })
            else:
                model = clone(model_factory)
        else:
            model = model_factory()

        model.fit(X_tr, y[train_idx])
        y_pred = model.predict(X_te)

        metrics['accuracy'].append(accuracy_score(y[test_idx], y_pred))
        metrics['precision'].append(precision_score(y[test_idx], y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y[test_idx], y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y[test_idx], y_pred, zero_division=0))

        try:
            proba = model.predict_proba(X_te)
            if proba.shape[1] == 2:
                proba = proba[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                metrics['auc'].append(roc_auc_score(y[test_idx], proba))
            else:
                metrics['auc'].append(0.0)
        except Exception:
            metrics['auc'].append(0.0)

    result = {}
    for k, v in metrics.items():
        result[f'{k}_mean'] = np.mean(v)
        result[f'{k}_std'] = np.std(v)
    return result


def run_grid_experiment(grid, dataset):
    """Run DL + baseline experiments for a single grid across all time windows."""
    dl_models = get_dl_models()
    baseline_models = get_baseline_models()
    all_models = {**dl_models, **baseline_models}

    labels = dataset.labels
    results = []

    for tw in TIME_WINDOWS:
        print(f"\n  Time window: {tw:.2f}s")

        # Filter events with enough post-clearing data
        valid_indices = []
        for i, event in enumerate(dataset.events):
            if event.clearing_time is not None:
                available = float(event.time[-1]) - event.clearing_time
                if available >= tw:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        if len(valid_indices) < 10:
            print(f"    Skipping: only {len(valid_indices)} events")
            continue

        filtered_events = [dataset.events[i] for i in valid_indices]
        filtered_labels = np.array([labels[i] for i in valid_indices])

        print(f"    Using {len(valid_indices)}/{len(dataset.events)} events")

        # Extract features
        extractor = TemporalFeatureExtractor(time_window=tw, relative_to_clearing=True)
        all_features = []
        for event in filtered_events:
            feat_dict = extractor.extract(event)
            all_features.append(feat_dict)

        feature_names = sorted(all_features[0].keys())
        X = np.array([
            [feat_dict.get(name, 0.0) for name in feature_names]
            for feat_dict in all_features
        ])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for model_name, model in all_models.items():
            t0 = time.time()
            print(f"    {model_name}...", end=' ', flush=True)

            try:
                cv_result = cv_evaluate(X, filtered_labels, model_name, model, N_SPLITS)
                elapsed = time.time() - t0

                row = {
                    'time_window': tw,
                    'model': model_name,
                    'model_type': 'deep_learning' if model_name in dl_models else 'classical',
                    'n_features': X.shape[1],
                    'n_events': len(filtered_events),
                    'training_time': elapsed,
                }
                row.update(cv_result)
                results.append(row)

                print(f"Acc={cv_result['accuracy_mean']:.4f} "
                      f"AUC={cv_result['auc_mean']:.4f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    'time_window': tw, 'model': model_name,
                    'model_type': 'deep_learning' if model_name in dl_models else 'classical',
                    'n_features': X.shape[1], 'n_events': len(filtered_events),
                    'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                    'auc_mean': 0.0, 'auc_std': 0.0,
                    'precision_mean': 0.0, 'precision_std': 0.0,
                    'recall_mean': 0.0, 'recall_std': 0.0,
                    'f1_mean': 0.0, 'f1_std': 0.0,
                    'training_time': 0.0,
                    'error': str(e),
                })

    return pd.DataFrame(results)


def plot_dl_comparison(results_df, grid, n_events):
    """Plot DL vs classical comparison across time windows."""
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 4.5))

    models = results_df['model'].unique()
    cmap = {'LSTM': '#e41a1c', 'TCN': '#377eb8', 'Transformer': '#4daf4a',
            'ETrees': '#984ea3', 'RF': '#ff7f00'}
    markers = {'LSTM': 'o', 'TCN': 's', 'Transformer': '^', 'ETrees': 'D', 'RF': 'v'}

    for model_name in models:
        mdf = results_df[results_df['model'] == model_name].sort_values('time_window')
        x = mdf['time_window'].values
        y = mdf['accuracy_mean'].values
        ys = mdf['accuracy_std'].values

        color = cmap.get(model_name, 'gray')
        marker = markers.get(model_name, 'o')
        lw = 2.0 if model_name in ('LSTM', 'TCN', 'Transformer') else 1.2
        ls = '-' if model_name in ('LSTM', 'TCN', 'Transformer') else '--'

        label_suffix = ' (DL)' if model_name in ('LSTM', 'TCN', 'Transformer') else ' (ML)'
        ax.plot(x, y, marker=marker, color=color, linewidth=lw, linestyle=ls,
                markersize=5, label=f'{model_name}{label_suffix}')
        ax.fill_between(x, y - ys, y + ys, alpha=0.1, color=color)

    ax.set_xlabel('Observation Time Window (s)')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.4, top=1.05)
    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Deep Learning vs Classical ML — {grid_label} (n={n_events})')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.grid(True, alpha=0.3)

    path = os.path.join(IMAGES_DIR, f'dl_vs_classical_{grid}_n{n_events}.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dl_training_time(results_df, grid, n_events):
    """Plot training time comparison."""
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    # Use the largest time window for comparison
    max_tw = results_df['time_window'].max()
    tw_df = results_df[results_df['time_window'] == max_tw].copy()

    if len(tw_df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # Bar chart: accuracy comparison at max time window
    colors = ['#e41a1c' if t == 'deep_learning' else '#984ea3'
              for t in tw_df['model_type']]
    ax1.barh(tw_df['model'], tw_df['accuracy_mean'], xerr=tw_df['accuracy_std'],
             color=colors, edgecolor='black', linewidth=0.5, capsize=3)
    ax1.set_xlabel('Accuracy')
    ax1.set_xlim(0.8, 1.02)
    ax1.set_title(f'Accuracy at t={max_tw:.1f}s')

    # Bar chart: training time
    ax2.barh(tw_df['model'], tw_df['training_time'],
             color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Training Time (s)')
    ax2.set_title('Training Time (5-fold CV)')

    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    fig.suptitle(f'{grid_label} (n={n_events})', fontsize=10)
    fig.tight_layout()

    path = os.path.join(IMAGES_DIR, f'dl_training_time_{grid}_n{n_events}.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    # ---- NineBusSystem ----
    print("\n=== Loading NineBusSystem data ===")
    t0 = time.time()
    nine_dataset = load_temporal_dataset('NineBusSystem', n_events=N_EVENTS, verbose=True)
    print(f"Loaded in {time.time() - t0:.1f}s — {nine_dataset}")

    print("\n=== Running NineBusSystem DL experiment ===")
    nine_results = run_grid_experiment('NineBusSystem', nine_dataset)

    nine_csv = os.path.join(OUTPUT_DIR, f'dl_results_NineBusSystem_n{N_EVENTS}.csv')
    nine_results.to_csv(nine_csv, index=False)
    print(f"\nSaved: {nine_csv}")

    plot_dl_comparison(nine_results, 'NineBusSystem', N_EVENTS)
    plot_dl_training_time(nine_results, 'NineBusSystem', N_EVENTS)

    # ---- NewEngland ----
    print("\n=== Loading NewEngland data ===")
    t0 = time.time()
    ne_dataset = load_temporal_dataset('NewEngland', n_events=N_EVENTS, verbose=True)
    print(f"Loaded in {time.time() - t0:.1f}s — {ne_dataset}")

    print("\n=== Running NewEngland DL experiment ===")
    ne_results = run_grid_experiment('NewEngland', ne_dataset)

    ne_csv = os.path.join(OUTPUT_DIR, f'dl_results_NewEngland_n{N_EVENTS}.csv')
    ne_results.to_csv(ne_csv, index=False)
    print(f"\nSaved: {ne_csv}")

    plot_dl_comparison(ne_results, 'NewEngland', N_EVENTS)
    plot_dl_training_time(ne_results, 'NewEngland', N_EVENTS)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY: Deep Learning vs Classical ML")
    print("=" * 60)

    for grid, results in [('NineBusSystem', nine_results), ('NewEngland', ne_results)]:
        grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
        print(f"\n{grid_label}:")
        print(f"{'Window':>8s} | {'LSTM':>10s} | {'TCN':>10s} | {'Transf.':>10s} | {'ETrees':>10s} | {'RF':>10s}")
        print("-" * 65)

        for tw in TIME_WINDOWS:
            tw_df = results[results['time_window'] == tw]
            if len(tw_df) == 0:
                continue
            vals = {}
            for m in ['LSTM', 'TCN', 'Transformer', 'ETrees', 'RF']:
                mdf = tw_df[tw_df['model'] == m]
                if len(mdf) > 0:
                    vals[m] = f"{mdf['accuracy_mean'].values[0]:.3f}"
                else:
                    vals[m] = "  N/A "
            print(f"  {tw:5.2f}s | {vals['LSTM']:>10s} | {vals['TCN']:>10s} | "
                  f"{vals.get('Transformer', 'N/A'):>10s} | {vals['ETrees']:>10s} | {vals['RF']:>10s}")

    print("\nAll done!")


if __name__ == '__main__':
    main()
