#!/usr/bin/env python3
"""Phase 5: Scale-Up Experiment — n=500 events, 17 classifiers, 8 time windows.

Runs the full early prediction experiment and temporal-vs-snapshot comparison
for the IEEE Access TSA paper.

Usage:
    python scripts/phase5_experiment.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Try optional boosting libraries
try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False

from tranpy.datasets.temporal_loaders import (
    load_temporal_dataset,
    list_available_events,
)
from tranpy.features.temporal import TemporalFeatureExtractor
from tranpy.features.early_prediction import run_early_prediction_experiment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Configuration ----
N_EVENTS = 500
TIME_WINDOWS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'papers', 'ieeaccess25')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'Images', 'temporal')

GRIDS = ['NineBusSystem', 'NewEngland']


def get_all_classifiers():
    """Return all 17 classifiers from the original paper."""
    classifiers = {
        'ETrees': ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
        'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'GP': GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=RANDOM_STATE),
        'Ada': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE, algorithm='SAMME'),
        'LR': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Ridge': RidgeClassifier(random_state=RANDOM_STATE),
        'SVM-R': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'SVM-L': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
        'NB': GaussianNB(),
        'Dummy': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
    }
    if _LGBM:
        classifiers['LGBM'] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
    return classifiers


def run_snapshot_comparison(grid, dataset, classifiers, n_splits=5):
    """Compare temporal features (10s window) vs original snapshot features.

    Returns a DataFrame with model, temporal_acc, snapshot_acc columns.
    """
    from tranpy.datasets import load_ieee9bus, load_newengland

    print(f"\n{'='*60}")
    print(f"SNAPSHOT vs TEMPORAL COMPARISON — {grid}")
    print(f"{'='*60}")

    # --- Temporal features at full 10s window ---
    print("Extracting temporal features (10s window)...")
    extractor = TemporalFeatureExtractor(time_window=10.0)
    X_temp, feat_names_temp = extractor.transform(dataset)
    X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=0.0, neginf=0.0)
    y_temp = dataset.labels
    print(f"  Temporal: {X_temp.shape[0]} samples x {X_temp.shape[1]} features")

    # --- Snapshot features ---
    print("Loading snapshot features...")
    if grid == 'NineBusSystem':
        snap_data = load_ieee9bus()
    else:
        snap_data = load_newengland()
    X_snap = snap_data.data
    y_snap = snap_data.target
    print(f"  Snapshot: {X_snap.shape[0]} samples x {X_snap.shape[1]} features")

    results = []
    for model_name, clf_template in classifiers.items():
        print(f"  Evaluating {model_name}...")
        row = {'model': model_name}

        # Temporal CV
        acc_temp = _cv_accuracy(X_temp, y_temp, clf_template, n_splits)
        row['temporal_accuracy'] = acc_temp['mean']
        row['temporal_accuracy_std'] = acc_temp['std']
        row['temporal_n_features'] = X_temp.shape[1]
        row['temporal_n_samples'] = X_temp.shape[0]

        # Snapshot CV (use same n_splits)
        acc_snap = _cv_accuracy(X_snap, y_snap, clf_template, n_splits)
        row['snapshot_accuracy'] = acc_snap['mean']
        row['snapshot_accuracy_std'] = acc_snap['std']
        row['snapshot_n_features'] = X_snap.shape[1]
        row['snapshot_n_samples'] = X_snap.shape[0]

        row['improvement'] = acc_temp['mean'] - acc_snap['mean']

        results.append(row)
        print(f"    Temporal: {acc_temp['mean']:.4f} | Snapshot: {acc_snap['mean']:.4f} | "
              f"Delta: {row['improvement']:+.4f}")

    return pd.DataFrame(results)


def _cv_accuracy(X, y, clf_template, n_splits):
    """Run CV and return mean/std accuracy."""
    from sklearn.base import clone

    min_class = min(np.bincount(y.astype(int)))
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        actual_splits = 2

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(clf_template)
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))

    return {'mean': np.mean(accs), 'std': np.std(accs)}


def _col(df, candidates):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def plot_early_prediction(results_df, grid, save_dir):
    """Generate publication-quality early prediction curve."""
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    model_col = _col(results_df, ['classifier', 'model'])
    acc_col = _col(results_df, ['accuracy_mean', 'accuracy'])
    acc_std_col = _col(results_df, ['accuracy_std'])

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 4.5))

    models = results_df[model_col].unique()
    # Sort by final accuracy (best on top in legend)
    final_accs = {}
    for m in models:
        mdf = results_df[results_df[model_col] == m]
        final_accs[m] = mdf[mdf['time_window'] == mdf['time_window'].max()][acc_col].values[0]
    models_sorted = sorted(models, key=lambda m: -final_accs[m])

    # Use distinct colors and markers
    cmap = plt.cm.tab20
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*', 'X', 'P', '8', 'd', '+', 'x', '1']

    for i, model_name in enumerate(models_sorted):
        model_df = results_df[results_df[model_col] == model_name].sort_values('time_window')
        x = model_df['time_window'].values
        y = model_df[acc_col].values
        y_std = model_df[acc_std_col].values

        color = cmap(i / len(models_sorted))
        marker = markers[i % len(markers)]

        ax.plot(x, y, marker=marker, color=color, linewidth=1.0, markersize=4,
                label=f'{model_name} ({final_accs[model_name]:.1%})')
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.08, color=color)

    ax.set_xlabel('Observation Time Window (s)')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.3, top=1.05)

    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Early Prediction Accuracy — {grid_label} (n={N_EVENTS})')

    ax.legend(loc='lower right', ncol=3, fontsize=6.5, framealpha=0.9)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    save_path = os.path.join(save_dir, f'early_prediction_{grid}_n{N_EVENTS}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison_bar(comparison_df, grid, save_dir):
    """Generate temporal-vs-snapshot bar chart."""
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    # Sort by temporal accuracy
    df = comparison_df.sort_values('temporal_accuracy', ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 5))

    y_pos = np.arange(len(df))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height / 2, df['snapshot_accuracy'], bar_height,
                     label=f"Snapshot ({df['snapshot_n_features'].iloc[0]} feat.)",
                     color='#92c5de', edgecolor='#2166ac', linewidth=0.5)
    bars2 = ax.barh(y_pos + bar_height / 2, df['temporal_accuracy'], bar_height,
                     label=f"Temporal ({df['temporal_n_features'].iloc[0]} feat.)",
                     color='#f4a582', edgecolor='#b2182b', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['model'] if 'model' in df.columns else df['classifier'], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0.4, 1.05)
    ax.legend(loc='lower right')

    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Snapshot vs Temporal Features — {grid_label}')

    save_path = os.path.join(save_dir, f'snapshot_vs_temporal_{grid}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_importance(dataset, grid, save_dir):
    """Generate feature importance heatmap using ETrees at each time window."""
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    windows = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    importance_by_window = {}

    for tw in windows:
        extractor = TemporalFeatureExtractor(time_window=tw)
        X, feat_names = extractor.transform(dataset)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = dataset.labels

        clf = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf.fit(X_scaled, y)

        importance_by_window[tw] = pd.Series(clf.feature_importances_, index=feat_names)

    # Build heatmap
    top_n = 20
    all_features = set()
    for imp in importance_by_window.values():
        all_features.update(imp.nlargest(top_n).index)

    avg_imp = {}
    for feat in all_features:
        avg_imp[feat] = np.mean([importance_by_window[w].get(feat, 0) for w in windows])
    top_features = sorted(avg_imp.keys(), key=lambda f: -avg_imp[f])[:top_n]

    matrix = np.zeros((len(top_features), len(windows)))
    for j, w in enumerate(windows):
        for i, feat in enumerate(top_features):
            matrix[i, j] = importance_by_window[w].get(feat, 0)

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 5.5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f'{w:.1f}s' for w in windows], rotation=45)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=6.5)
    ax.set_xlabel('Observation Window')
    ax.set_ylabel('Feature')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Importance (ETrees)')

    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Feature Importance — {grid_label} (n={N_EVENTS})')

    save_path = os.path.join(save_dir, f'feature_importance_{grid}_n{N_EVENTS}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    classifiers = get_all_classifiers()
    print(f"Classifiers: {list(classifiers.keys())} ({len(classifiers)} total)")
    print(f"Time windows: {TIME_WINDOWS}")
    print(f"Events per grid: {N_EVENTS}")

    all_results = {}
    all_comparisons = {}

    for grid in GRIDS:
        print(f"\n{'='*60}")
        print(f"LOADING DATA — {grid}")
        print(f"{'='*60}")

        t0 = time.time()
        dataset = load_temporal_dataset(grid, n_events=N_EVENTS, verbose=True)
        load_time = time.time() - t0
        print(f"Loading took {load_time:.1f}s")
        print(f"Dataset: {dataset}")

        # ---- Early Prediction Experiment ----
        print(f"\n{'='*60}")
        print(f"EARLY PREDICTION — {grid}")
        print(f"{'='*60}")

        t0 = time.time()
        results = run_early_prediction_experiment(
            dataset=dataset,
            classifiers=classifiers,
            time_windows=TIME_WINDOWS,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            verbose=True,
        )
        exp_time = time.time() - t0
        print(f"\nExperiment took {exp_time:.1f}s")

        # Save results CSV
        csv_path = os.path.join(OUTPUT_DIR, f'temporal_results_{grid}_n{N_EVENTS}.csv')
        results.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        all_results[grid] = results

        # ---- Generate Figures ----
        print(f"\nGenerating figures for {grid}...")
        plot_early_prediction(results, grid, IMAGES_DIR)
        plot_feature_importance(dataset, grid, IMAGES_DIR)

        # ---- Snapshot Comparison ----
        comparison = run_snapshot_comparison(grid, dataset, classifiers, n_splits=N_SPLITS)
        comp_csv = os.path.join(OUTPUT_DIR, f'snapshot_vs_temporal_{grid}.csv')
        comparison.to_csv(comp_csv, index=False)
        print(f"Saved: {comp_csv}")
        all_comparisons[grid] = comparison

        plot_comparison_bar(comparison, grid, IMAGES_DIR)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for grid in GRIDS:
        results = all_results[grid]
        comparison = all_comparisons[grid]

        grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
        print(f"\n--- {grid_label} ---")

        # Best model at each time window
        model_col = _col(results, ['classifier', 'model'])
        acc_col = _col(results, ['accuracy_mean', 'accuracy'])
        for tw in TIME_WINDOWS:
            tw_df = results[results['time_window'] == tw]
            if len(tw_df) == 0:
                continue
            best = tw_df.loc[tw_df[acc_col].idxmax()]
            print(f"  t={tw:5.2f}s: Best={best[model_col]:6s} Acc={best[acc_col]:.4f}")

        # Snapshot vs temporal
        print(f"\n  Snapshot vs Temporal (10s window):")
        for _, row in comparison.iterrows():
            print(f"    {row['model']:6s}: Snap={row['snapshot_accuracy']:.4f} "
                  f"Temp={row['temporal_accuracy']:.4f} "
                  f"Delta={row['improvement']:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
