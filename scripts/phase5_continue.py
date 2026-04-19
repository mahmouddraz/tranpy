#!/usr/bin/env python3
"""Phase 5 continuation: plot NineBusSystem results, run NewEngland, comparison.

NineBusSystem early prediction CSV already saved. This script:
1. Plots NineBusSystem results
2. Runs NewEngland early prediction
3. Runs snapshot-vs-temporal comparison for both grids
4. Generates all figures
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
from sklearn.metrics import accuracy_score
from sklearn.base import clone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tranpy.datasets.temporal_loaders import load_temporal_dataset
from tranpy.features.temporal import TemporalFeatureExtractor
from tranpy.features.early_prediction import run_early_prediction_experiment

N_EVENTS = 500
TIME_WINDOWS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'papers', 'ieeaccess25')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'Images', 'temporal')


def get_classifiers():
    return {
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


def _col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in columns")


def cv_accuracy(X, y, clf_template, n_splits=5):
    min_class = min(np.bincount(y.astype(int)))
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        actual_splits = 2
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = clone(clf_template)
        clf.fit(X_tr, y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X_te)))
    return np.mean(accs), np.std(accs)


def plot_early_prediction(results_df, grid, n_events):
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    model_col = _col(results_df, ['classifier', 'model'])
    acc_col = _col(results_df, ['accuracy_mean', 'accuracy'])
    acc_std_col = 'accuracy_std'

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 4.5))
    models = results_df[model_col].unique()

    final_accs = {}
    for m in models:
        mdf = results_df[results_df[model_col] == m]
        final_accs[m] = mdf[mdf['time_window'] == mdf['time_window'].max()][acc_col].values[0]
    models_sorted = sorted(models, key=lambda m: -final_accs[m])

    # Exclude Dummy from plot (clutters)
    models_sorted = [m for m in models_sorted if m != 'Dummy']

    cmap = plt.cm.tab20
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*', 'X', 'P', '8', 'd', '+', 'x']

    for i, mn in enumerate(models_sorted):
        mdf = results_df[results_df[model_col] == mn].sort_values('time_window')
        x, y, ys = mdf['time_window'].values, mdf[acc_col].values, mdf[acc_std_col].values
        color = cmap(i / max(len(models_sorted), 1))
        marker = markers[i % len(markers)]
        ax.plot(x, y, marker=marker, color=color, linewidth=1.0, markersize=4,
                label=f'{mn} ({final_accs[mn]:.1%})')
        ax.fill_between(x, y - ys, y + ys, alpha=0.08, color=color)

    ax.set_xlabel('Observation Time Window (s)')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.4, top=1.05)
    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Early Prediction Accuracy — {grid_label} (n={n_events})')
    ax.legend(loc='lower right', ncol=3, fontsize=6, framealpha=0.9)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    path = os.path.join(IMAGES_DIR, f'early_prediction_{grid}_n{n_events}.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison_bar(comp_df, grid):
    from tranpy.visualization.temporal import set_ieee_style
    set_ieee_style()

    df = comp_df.sort_values('temporal_accuracy', ascending=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.16, 5))
    y_pos = np.arange(len(df))
    bh = 0.35

    ax.barh(y_pos - bh / 2, df['snapshot_accuracy'], bh,
            label=f"Snapshot ({int(df['snapshot_n_features'].iloc[0])} feat.)",
            color='#92c5de', edgecolor='#2166ac', linewidth=0.5)
    ax.barh(y_pos + bh / 2, df['temporal_accuracy'], bh,
            label=f"Temporal ({int(df['temporal_n_features'].iloc[0])} feat.)",
            color='#f4a582', edgecolor='#b2182b', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['classifier'] if 'classifier' in df.columns else df['model'], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0.4, 1.05)
    ax.legend(loc='lower right')
    grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
    ax.set_title(f'Snapshot vs Temporal Features — {grid_label}')

    path = os.path.join(IMAGES_DIR, f'snapshot_vs_temporal_{grid}.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(dataset, grid):
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
        clf.fit(scaler.fit_transform(X), y)
        importance_by_window[tw] = pd.Series(clf.feature_importances_, index=feat_names)

    top_n = 20
    all_feats = set()
    for imp in importance_by_window.values():
        all_feats.update(imp.nlargest(top_n).index)
    avg = {f: np.mean([importance_by_window[w].get(f, 0) for w in windows]) for f in all_feats}
    top_features = sorted(avg, key=lambda f: -avg[f])[:top_n]

    matrix = np.zeros((len(top_features), len(windows)))
    for j, w in enumerate(windows):
        for i, f in enumerate(top_features):
            matrix[i, j] = importance_by_window[w].get(f, 0)

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

    path = os.path.join(IMAGES_DIR, f'feature_importance_{grid}_n{N_EVENTS}.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def run_snapshot_comparison(grid, dataset, classifiers):
    from tranpy.datasets import load_ieee9bus, load_newengland

    print(f"\n--- Snapshot vs Temporal Comparison: {grid} ---")

    # Temporal features at 10s
    print("  Extracting temporal features (10s)...")
    extractor = TemporalFeatureExtractor(time_window=10.0)
    X_temp, _ = extractor.transform(dataset)
    X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=0.0, neginf=0.0)
    y_temp = dataset.labels
    print(f"  Temporal: {X_temp.shape}")

    # Snapshot features
    snap = load_ieee9bus() if grid == 'NineBusSystem' else load_newengland()
    X_snap, y_snap = snap.data, snap.target
    print(f"  Snapshot: {X_snap.shape}")

    rows = []
    for name, clf in classifiers.items():
        print(f"    {name}...", end=' ')
        t_mean, t_std = cv_accuracy(X_temp, y_temp, clf, N_SPLITS)
        s_mean, s_std = cv_accuracy(X_snap, y_snap, clf, N_SPLITS)
        print(f"Temporal={t_mean:.4f} Snapshot={s_mean:.4f} Delta={t_mean - s_mean:+.4f}")
        rows.append({
            'classifier': name,
            'temporal_accuracy': t_mean, 'temporal_accuracy_std': t_std,
            'temporal_n_features': X_temp.shape[1], 'temporal_n_samples': X_temp.shape[0],
            'snapshot_accuracy': s_mean, 'snapshot_accuracy_std': s_std,
            'snapshot_n_features': X_snap.shape[1], 'snapshot_n_samples': X_snap.shape[0],
            'improvement': t_mean - s_mean,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    classifiers = get_classifiers()

    # ---- Step 1: Plot NineBusSystem (already computed) ----
    nine_csv = os.path.join(OUTPUT_DIR, 'temporal_results_NineBusSystem_n500.csv')
    if os.path.exists(nine_csv):
        print("=== Plotting NineBusSystem results ===")
        nine_results = pd.read_csv(nine_csv)
        plot_early_prediction(nine_results, 'NineBusSystem', N_EVENTS)
    else:
        print(f"WARNING: {nine_csv} not found — skipping NineBusSystem plots")

    # ---- Step 2: Run NewEngland experiment ----
    print("\n=== Loading NewEngland data ===")
    t0 = time.time()
    ne_dataset = load_temporal_dataset('NewEngland', n_events=N_EVENTS, verbose=True)
    print(f"Loaded in {time.time() - t0:.1f}s — {ne_dataset}")

    print("\n=== Running NewEngland early prediction ===")
    t0 = time.time()
    ne_results = run_early_prediction_experiment(
        dataset=ne_dataset,
        classifiers=classifiers,
        time_windows=TIME_WINDOWS,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    print(f"Experiment took {time.time() - t0:.1f}s")

    ne_csv = os.path.join(OUTPUT_DIR, f'temporal_results_NewEngland_n{N_EVENTS}.csv')
    ne_results.to_csv(ne_csv, index=False)
    print(f"Saved: {ne_csv}")

    plot_early_prediction(ne_results, 'NewEngland', N_EVENTS)

    # ---- Step 3: Feature importance for both grids ----
    print("\n=== Feature importance plots ===")

    # Load NineBus data for feature importance & comparison
    print("Loading NineBusSystem for feature importance...")
    nine_dataset = load_temporal_dataset('NineBusSystem', n_events=N_EVENTS, verbose=True)

    print("NineBusSystem feature importance...")
    plot_feature_importance(nine_dataset, 'NineBusSystem')
    print("NewEngland feature importance...")
    plot_feature_importance(ne_dataset, 'NewEngland')

    # ---- Step 4: Snapshot vs Temporal Comparison ----
    print("\n=== Snapshot vs Temporal Comparison ===")

    for grid, dataset in [('NineBusSystem', nine_dataset), ('NewEngland', ne_dataset)]:
        comp = run_snapshot_comparison(grid, dataset, classifiers)
        comp_csv = os.path.join(OUTPUT_DIR, f'snapshot_vs_temporal_{grid}.csv')
        comp.to_csv(comp_csv, index=False)
        print(f"Saved: {comp_csv}")
        plot_comparison_bar(comp, grid)

    # ---- Summary ----
    print("\n=== FINAL SUMMARY ===")
    for grid in ['NineBusSystem', 'NewEngland']:
        csv_path = os.path.join(OUTPUT_DIR, f'temporal_results_{grid}_n{N_EVENTS}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        model_col = _col(df, ['classifier', 'model'])
        acc_col = _col(df, ['accuracy_mean', 'accuracy'])

        grid_label = 'IEEE 9-Bus' if grid == 'NineBusSystem' else 'New England 39-Bus'
        print(f"\n{grid_label}:")
        for tw in TIME_WINDOWS:
            tw_df = df[df['time_window'] == tw]
            if len(tw_df) == 0:
                continue
            best = tw_df.loc[tw_df[acc_col].idxmax()]
            print(f"  t={tw:5.2f}s: Best={best[model_col]:6s} Acc={best[acc_col]:.4f}")

        comp_csv = os.path.join(OUTPUT_DIR, f'snapshot_vs_temporal_{grid}.csv')
        if os.path.exists(comp_csv):
            comp = pd.read_csv(comp_csv)
            print(f"  Avg temporal improvement: {comp['improvement'].mean():+.4f}")

    print("\nAll done!")


if __name__ == '__main__':
    main()
