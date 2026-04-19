"""Publication-quality visualizations for temporal trajectory analysis.

Generates figures suitable for IEEE Access and other power systems journals.
All plots use consistent styling with configurable output formats.

Plots are fault-timing-aware: vertical lines mark fault inception and
clearing times per event, and time axes can be shown relative to clearing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from ..datasets.temporal_loaders import (
    TemporalEventData,
    TemporalDataset,
    get_generator_columns,
    get_bus_columns,
)


# IEEE-friendly color palette
COLORS = {
    'stable': '#2166ac',       # blue
    'unstable': '#b2182b',     # red
    'stable_light': '#92c5de',
    'unstable_light': '#f4a582',
    'neutral': '#666666',
    'accent': '#1b7837',
    'fault': '#ff7f00',        # orange for fault markers
    'clearing': '#4daf4a',     # green for clearing markers
}

# Default figure sizes for IEEE two-column format
FIG_SINGLE_COL = (3.5, 2.8)   # single column
FIG_DOUBLE_COL = (7.16, 3.5)  # double column
FIG_SQUARE = (3.5, 3.5)


def set_ieee_style():
    """Set matplotlib parameters for IEEE publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
    })


def _add_fault_markers(ax, event: TemporalEventData, alpha: float = 0.7):
    """Add vertical lines for fault inception and clearing on an axes."""
    if event.fault_time is not None:
        ax.axvline(
            x=event.fault_time, color=COLORS['fault'],
            linestyle='--', linewidth=0.8, alpha=alpha,
        )
    if event.clearing_time is not None:
        ax.axvline(
            x=event.clearing_time, color=COLORS['clearing'],
            linestyle='--', linewidth=0.8, alpha=alpha,
        )


def _add_fault_legend_entries() -> List[Line2D]:
    """Return legend handles for fault/clearing markers."""
    return [
        Line2D([0], [0], color=COLORS['fault'], linestyle='--',
               linewidth=0.8, label='Fault'),
        Line2D([0], [0], color=COLORS['clearing'], linestyle='--',
               linewidth=0.8, label='Clearing'),
    ]


def plot_trajectory_comparison(
    dataset: TemporalDataset,
    variable: str = 'c:firot',
    generator: Optional[str] = None,
    relative_time: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot overlaid trajectories comparing stable vs unstable events.

    Creates a figure showing all stable event trajectories in blue and
    all unstable event trajectories in red.

    Args:
        dataset: TemporalDataset with labeled events
        variable: Variable pattern to plot (e.g., 'c:firot', 's:xspeed')
        generator: Specific generator to plot (None = first found)
        relative_time: If True, shift time so t=0 is fault clearing
        figsize: Figure size tuple (width, height) in inches
        title: Plot title (auto-generated if None)
        save_path: Path to save figure (None = don't save)
        ax: Existing axes to plot on (None = create new figure)

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    stable_events = dataset.get_stable_events()
    unstable_events = dataset.get_unstable_events()

    for events, color, light_color, label in [
        (stable_events, COLORS['stable'], COLORS['stable_light'], 'Stable'),
        (unstable_events, COLORS['unstable'], COLORS['unstable_light'], 'Unstable'),
    ]:
        for i, event in enumerate(events):
            gen_cols = get_generator_columns(event.trajectory)
            if generator:
                target_gen = generator
            else:
                target_gen = list(gen_cols.keys())[0]

            col = _find_variable_col(gen_cols.get(target_gen, []), variable)
            if col is None:
                continue

            y = event.trajectory[col].values
            t = event.time.copy()

            if relative_time and event.clearing_time is not None:
                t = t - event.clearing_time

            ax.plot(
                t, y,
                color=light_color,
                alpha=0.6,
                linewidth=0.8,
                label=label if i == 0 else None,
            )

    if relative_time:
        ax.set_xlabel('Time After Clearing (s)')
        ax.axvline(x=0, color=COLORS['clearing'], linestyle='--',
                   linewidth=0.8, alpha=0.7, label='Clearing')
    else:
        ax.set_xlabel('Time (s)')
        # Show median fault/clearing times as reference lines
        fault_times = [e.fault_time for e in dataset.events if e.fault_time is not None]
        clear_times = [e.clearing_time for e in dataset.events if e.clearing_time is not None]
        if fault_times:
            ax.axvspan(
                min(fault_times), max(fault_times) if clear_times else min(fault_times) + 0.5,
                alpha=0.08, color=COLORS['fault'], label='Fault range',
            )
        if clear_times:
            med_clear = np.median(clear_times)
            ax.axvline(x=med_clear, color=COLORS['clearing'], linestyle='--',
                       linewidth=0.8, alpha=0.7, label=f'Median clearing ({med_clear:.1f}s)')

    # Auto ylabel based on variable
    ylabel_map = {
        'c:firot': 'Rotor Angle (deg)',
        's:xspeed': 'Speed Deviation (p.u.)',
        'm:u': 'Voltage Magnitude (p.u.)',
        'm:fehz': 'Frequency (Hz)',
        's:P1': 'Active Power (MW)',
        's:Q1': 'Reactive Power (Mvar)',
    }
    ylabel = ylabel_map.get(variable, variable)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    else:
        gen_label = generator or 'Generator'
        ax.set_title(f'{gen_label} — {ylabel}')

    ax.legend(loc='best')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_phase_portrait(
    dataset: TemporalDataset,
    generator: Optional[str] = None,
    post_clearing_only: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot rotor angle vs speed deviation phase portraits.

    Stable events show closed orbits converging to equilibrium.
    Unstable events show diverging trajectories.

    Args:
        dataset: TemporalDataset with labeled events
        generator: Specific generator (None = first found)
        post_clearing_only: If True, only show post-clearing trajectory
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_SQUARE

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for event in dataset.events:
        gen_cols = get_generator_columns(event.trajectory)
        target_gen = generator or list(gen_cols.keys())[0]
        cols = gen_cols.get(target_gen, [])

        angle_col = _find_variable_col(cols, 'c:firot')
        speed_col = _find_variable_col(cols, 's:xspeed')

        if angle_col is None or speed_col is None:
            continue

        angle = event.trajectory[angle_col].values
        speed = event.trajectory[speed_col].values
        time = event.time

        # Optionally filter to post-clearing
        if post_clearing_only and event.clearing_time is not None:
            mask = time >= event.clearing_time
            angle = angle[mask]
            speed = speed[mask]

        color = COLORS['stable'] if event.label == 0 else COLORS['unstable']
        alpha = 0.7

        ax.plot(angle, speed, color=color, alpha=alpha, linewidth=0.8)

        # Mark start (clearing point) and end
        ax.plot(angle[0], speed[0], 'o', color=color, markersize=3, alpha=alpha)
        ax.plot(angle[-1], speed[-1], 's', color=color, markersize=3, alpha=alpha)

    ax.set_xlabel('Rotor Angle (deg)')
    ax.set_ylabel('Speed Deviation (p.u.)')

    if title:
        ax.set_title(title)
    else:
        gen_label = generator or 'Generator'
        suffix = ' (post-clearing)' if post_clearing_only else ''
        ax.set_title(f'Phase Portrait — {gen_label}{suffix}')

    legend_elements = [
        Line2D([0], [0], color=COLORS['stable'], linewidth=1.5, label='Stable'),
        Line2D([0], [0], color=COLORS['unstable'], linewidth=1.5, label='Unstable'),
        Line2D([0], [0], marker='o', color='gray', markersize=4,
               linestyle='', label='Clearing point'),
        Line2D([0], [0], marker='s', color='gray', markersize=4,
               linestyle='', label='End'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_voltage_heatmap(
    event: TemporalEventData,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot voltage evolution heatmap across all buses over time.

    Marks fault inception and clearing with vertical lines.

    Args:
        event: Single TemporalEventData object
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    bus_cols = get_bus_columns(event.trajectory)

    # Collect voltage data
    bus_names = []
    voltage_data = []
    for bus_id, cols in sorted(bus_cols.items()):
        v_col = _find_variable_col(cols, 'm:u in p.u.')
        if v_col:
            bus_names.append(bus_id)
            voltage_data.append(event.trajectory[v_col].values)

    if not voltage_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No voltage data found', transform=ax.transAxes, ha='center')
        return fig

    voltage_matrix = np.array(voltage_data)  # (n_buses, n_timesteps)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        voltage_matrix,
        aspect='auto',
        cmap='RdYlGn',
        vmin=0.8,
        vmax=1.1,
        extent=[event.time[0], event.time[-1], len(bus_names) - 0.5, -0.5],
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bus')
    ax.set_yticks(range(len(bus_names)))
    ax.set_yticklabels(bus_names, fontsize=6)

    # Mark fault and clearing times
    _add_fault_markers(ax, event, alpha=0.9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Voltage (p.u.)')

    label_str = 'Stable' if event.label == 0 else 'Unstable'
    fault_info = ''
    if event.fault_time is not None and event.clearing_time is not None:
        fault_info = f', fault={event.fault_duration * 1000:.0f}ms'
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Voltage Evolution — Event {event.event_id} ({label_str}{fault_info})')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_early_prediction_curve(
    results_df: pd.DataFrame,
    metric: str = 'accuracy',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot classifier performance vs post-clearing observation time.

    Args:
        results_df: DataFrame from run_early_prediction_experiment
        metric: Metric to plot ('accuracy', 'precision', 'recall', 'f1', 'auc')
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Support both old ('model'/'accuracy') and new ('classifier'/'accuracy_mean') column names
    model_col = 'classifier' if 'classifier' in results_df.columns else 'model'
    metric_col = f'{metric}_mean' if f'{metric}_mean' in results_df.columns else metric
    std_col = f'{metric}_std'

    models = results_df[model_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model_name, color in zip(models, colors):
        model_df = results_df[results_df[model_col] == model_name].sort_values('time_window')

        x = model_df['time_window'].values
        y = model_df[metric_col].values
        y_std = model_df[std_col].values

        ax.plot(x, y, 'o-', color=color, linewidth=1.2, markersize=4, label=model_name)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.15, color=color)

    ax.set_xlabel('Post-Clearing Observation Time (s)')
    ax.set_ylabel(metric.capitalize())
    ax.set_xscale('log')

    # Auto y-axis: zoom in on the relevant range
    all_y = results_df[metric_col].values
    y_min = max(0.0, np.min(all_y) - 0.1)
    ax.set_ylim(bottom=y_min, top=1.02)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric.capitalize()} vs. Post-Clearing Observation Time')

    ax.legend(loc='lower right', ncol=2)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_feature_importance_heatmap(
    importance_by_window: Dict[float, pd.Series],
    top_n: int = 15,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot heatmap showing feature importance across post-clearing time windows.

    Args:
        importance_by_window: Dict mapping time_window to Series of feature importances
        top_n: Number of top features to show
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    # Build importance matrix
    windows = sorted(importance_by_window.keys())
    all_features = set()
    for imp in importance_by_window.values():
        all_features.update(imp.nlargest(top_n).index)

    # Sort features by average importance
    avg_importance = {}
    for feat in all_features:
        avg_importance[feat] = np.mean([
            importance_by_window[w].get(feat, 0) for w in windows
        ])

    top_features = sorted(avg_importance.keys(), key=lambda f: -avg_importance[f])[:top_n]

    # Build matrix
    matrix = np.zeros((len(top_features), len(windows)))
    for j, w in enumerate(windows):
        for i, feat in enumerate(top_features):
            matrix[i, j] = importance_by_window[w].get(feat, 0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f'{w:.1f}s' for w in windows], rotation=45)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=7)
    ax.set_xlabel('Post-Clearing Observation Window')
    ax.set_ylabel('Feature')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Importance')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Feature Importance vs. Post-Clearing Window')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_multi_generator_trajectories(
    event: TemporalEventData,
    variable: str = 'c:firot',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot trajectories of all generators in a single event.

    Marks fault inception and clearing with vertical lines.

    Args:
        event: Single TemporalEventData object
        variable: Variable to plot
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gen_cols = get_generator_columns(event.trajectory)
    colors = plt.cm.tab10(np.linspace(0, 1, len(gen_cols)))

    for (gen_id, cols), color in zip(gen_cols.items(), colors):
        col = _find_variable_col(cols, variable)
        if col is None:
            continue

        y = event.trajectory[col].values
        ax.plot(event.time, y, color=color, linewidth=0.8, label=gen_id)

    ax.set_xlabel('Time (s)')

    ylabel_map = {
        'c:firot': 'Rotor Angle (deg)',
        's:xspeed': 'Speed Deviation (p.u.)',
        's:P1': 'Active Power (MW)',
    }
    ax.set_ylabel(ylabel_map.get(variable, variable))

    # Mark fault and clearing times
    _add_fault_markers(ax, event)

    label_str = 'Stable' if event.label == 0 else 'Unstable'
    fault_info = ''
    if event.fault_time is not None and event.clearing_time is not None:
        fault_info = f', fault={event.fault_duration * 1000:.0f}ms'

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'All Generators — Event {event.event_id} ({label_str}{fault_info})')

    # Combine generator legend with fault markers
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(_add_fault_legend_entries())
    ax.legend(handles=handles, loc='best', fontsize=6, ncol=3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_fault_timing_distribution(
    dataset: TemporalDataset,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of fault and clearing times across events.

    Shows why absolute time windows are misleading and relative-to-clearing
    windows are necessary.

    Args:
        dataset: TemporalDataset with fault timing detected
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = FIG_DOUBLE_COL

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    stable = dataset.get_stable_events()
    unstable = dataset.get_unstable_events()

    # Panel 1: Fault inception time distribution
    ax = axes[0]
    for events, color, label in [
        (stable, COLORS['stable'], 'Stable'),
        (unstable, COLORS['unstable'], 'Unstable'),
    ]:
        ft = [e.fault_time for e in events if e.fault_time is not None]
        if ft:
            ax.hist(ft, bins=15, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_xlabel('Fault Inception Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Fault Inception')
    ax.legend(fontsize=7)

    # Panel 2: Fault clearing time distribution
    ax = axes[1]
    for events, color, label in [
        (stable, COLORS['stable'], 'Stable'),
        (unstable, COLORS['unstable'], 'Unstable'),
    ]:
        ct = [e.clearing_time for e in events if e.clearing_time is not None]
        if ct:
            ax.hist(ct, bins=15, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_xlabel('Fault Clearing Time (s)')
    ax.set_title('Fault Clearing')
    ax.legend(fontsize=7)

    # Panel 3: Fault duration distribution
    ax = axes[2]
    for events, color, label in [
        (stable, COLORS['stable'], 'Stable'),
        (unstable, COLORS['unstable'], 'Unstable'),
    ]:
        fd = [e.fault_duration * 1000 for e in events if e.fault_duration is not None]
        if fd:
            ax.hist(fd, bins=15, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_xlabel('Fault Duration (ms)')
    ax.set_title('Fault Duration')
    ax.legend(fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10)
    else:
        fig.suptitle(f'Fault Timing Distribution — {dataset.grid_name}', fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def _find_variable_col(cols: List[str], pattern: str) -> Optional[str]:
    """Find column matching a variable pattern."""
    pattern_lower = pattern.lower()
    for c in cols:
        if pattern_lower in c.lower():
            return c
    return None
