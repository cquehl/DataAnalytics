#!/usr/bin/env python3
"""
ANOVA Analysis Test Script
==========================

Tests whether factor scores (or raw features) differ significantly
across price tiers using one-way ANOVA with effect size calculations.

Parameters:
    data_file     - Path to input CSV
    suppliers     - List of suppliers to filter
    start_date    - Filter from date
    features      - List of features to analyze
    group_col     - Column to group by (default: TIER)
    top_n_groups  - Number of top groups to include

Outputs:
    - ANOVA results table (CSV)
    - Effect size visualization (PNG)
    - Group means table (CSV)
    - Text report (TXT)
"""

import sys
from pathlib import Path

# Hybrid import: works both when pip-installed and when run directly
try:
    from efa_core import data, stats, viz, output, config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from efa_core import data, stats, viz, output, config

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PARAMETERS
# =============================================================================
DEFAULTS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': config.DEFAULT_SUPPLIERS,
    'start_date': config.DEFAULT_START_DATE,
    'features': config.DEFAULT_EFA_FEATURES,
    'group_col': 'TIER',
    'top_n_groups': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}

TEST_NAME = 'anova'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_effect_sizes(anova_df, output_dir, test_name):
    """Create bar chart of effect sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [viz.get_colors()['significant'] if sig else viz.get_colors()['not_significant']
              for sig in anova_df['significant']]

    bars = ax.barh(anova_df['variable'], anova_df['eta_squared'], color=colors)

    # Add effect size threshold lines
    ax.axvline(x=0.01, color='orange', linestyle='--', alpha=0.7, label='Small (0.01)')
    ax.axvline(x=0.06, color='orange', linestyle='-', alpha=0.7, label='Medium (0.06)')
    ax.axvline(x=0.14, color='red', linestyle='-', alpha=0.7, label='Large (0.14)')

    ax.set_xlabel('Effect Size (eta-squared)')
    ax.set_title('ANOVA Effect Sizes by Variable\n(green = significant p<0.05)')
    ax.legend(loc='lower right')

    # Add value labels
    for bar, eta in zip(bars, anova_df['eta_squared']):
        ax.text(eta + 0.01, bar.get_y() + bar.get_height()/2,
                f'{eta:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'effect-sizes')


def plot_group_means(means_df, features, output_dir, test_name):
    """Create heatmap of standardized group means."""
    # Standardize for visualization
    plot_data = means_df[features].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(plot_data, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Mean Values by Group (darker = more extreme)')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Group')

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'group-means')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(params: dict) -> dict:
    """Run ANOVA analysis pipeline."""
    print("=" * 70)
    print("ANOVA ANALYSIS")
    print("=" * 70)

    # Setup output directory
    output_dir = output.get_output_dir(TEST_NAME, params['output_base'])

    # Load and prepare data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = data.load_csv(params['data_file'])

    if params['suppliers']:
        df = data.filter_by_suppliers(df, params['suppliers'])

    if params['start_date']:
        df = data.filter_by_date(df, params['start_date'])

    df, tier_map = data.create_tier_labels(df)

    # Filter to top groups
    group_col = params['group_col']
    top_groups = df[group_col].value_counts().head(params['top_n_groups']).index.tolist()
    df_filtered = df[df[group_col].isin(top_groups)].copy()

    print(f"\nAnalyzing top {params['top_n_groups']} groups ({len(df_filtered):,} records)")

    # Drop rows with missing features
    features = params['features']
    df_clean = df_filtered.dropna(subset=features)
    print(f"Records with complete data: {len(df_clean):,}")

    # Run ANOVA for each feature
    print("\n" + "=" * 70)
    print("ANOVA RESULTS")
    print("=" * 70)

    anova_df = stats.run_anova_multi(df_clean, features, group_col)
    stats.print_anova_results(anova_df)
    output.save_csv(anova_df, output_dir, TEST_NAME, 'results')

    # Calculate group means
    means_df = stats.calculate_group_means(df_clean, features, group_col)
    output.save_csv(means_df, output_dir, TEST_NAME, 'group-means', index=True)

    print("\nGroup Means:")
    print(means_df.to_string())

    # Create visualizations
    plot_effect_sizes(anova_df, output_dir, TEST_NAME)
    plot_group_means(means_df, features, output_dir, TEST_NAME)

    # Generate report
    report = generate_report(anova_df, means_df, params)
    output.save_report(report, output_dir, TEST_NAME)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'anova': anova_df,
        'means': means_df,
        'output_dir': output_dir,
    }


def generate_report(anova_df, means_df, params):
    """Generate text report."""
    lines = [
        "=" * 70,
        "ANOVA ANALYSIS REPORT",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Data file: {params['data_file']}",
        f"Group column: {params['group_col']}",
        f"Features analyzed: {len(params['features'])}",
        f"Top groups included: {params['top_n_groups']}",
        "",
        "RESULTS SUMMARY",
        "-" * 50,
    ]

    sig_features = anova_df[anova_df['significant']]
    nonsig_features = anova_df[~anova_df['significant']]

    lines.append(f"\nSignificant differences found: {len(sig_features)}/{len(anova_df)}")

    if len(sig_features) > 0:
        lines.append("\nFeatures with significant group differences:")
        for _, row in sig_features.iterrows():
            lines.append(f"  - {row['variable']}: F={row['f_statistic']:.2f}, eta^2={row['eta_squared']:.3f} ({row['effect_size']})")

    if len(nonsig_features) > 0:
        lines.append("\nFeatures without significant differences:")
        for _, row in nonsig_features.iterrows():
            lines.append(f"  - {row['variable']}: p={row['p_value']:.3f}")

    lines.extend([
        "",
        "GROUP MEANS",
        "-" * 50,
        means_df.to_string(),
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    params = {**DEFAULTS}
    results = run_analysis(params)
