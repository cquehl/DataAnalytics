#!/usr/bin/env python3
"""
Kolmogorov-Smirnov Test Script
==============================

Compares distributions of features across groups using KS tests.
Useful for checking if tier distributions are truly different.

Parameters:
    data_file     - Path to input CSV
    suppliers     - List of suppliers to filter
    start_date    - Filter from date
    features      - List of features to analyze
    group_col     - Column to group by (default: TIER)
    baseline_group - Group to compare against (default: largest)

Outputs:
    - KS test results (CSV)
    - Distribution comparison plots (PNG)
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
    'baseline_group': None,  # None = use largest group
    'top_n_groups': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}

TEST_NAME = 'ks-test'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_distributions(df, feature, group_col, output_dir, test_name):
    """Create overlapping distribution plot for a feature."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = df[group_col].unique()
    for group in sorted(groups):
        subset = df[df[group_col] == group][feature].dropna()
        sns.kdeplot(subset, label=group, ax=ax)

    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {feature} by {group_col}')
    ax.legend()

    output.save_figure(fig, output_dir, test_name, f'dist-{feature.lower()}')


def plot_ks_summary(ks_results_all, output_dir, test_name):
    """Create summary plot of KS statistics."""
    # Pivot for heatmap
    pivot = ks_results_all.pivot(index='comparison', columns='feature', values='ks_statistic')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
    ax.set_title('KS Statistics by Feature and Comparison\n(higher = more different)')

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'ks-summary')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(params: dict) -> dict:
    """Run KS test analysis pipeline."""
    print("=" * 70)
    print("KOLMOGOROV-SMIRNOV TEST ANALYSIS")
    print("=" * 70)

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

    # Determine baseline
    baseline = params['baseline_group'] or df_filtered[group_col].value_counts().index[0]
    print(f"Baseline group: {baseline}")

    # Run KS tests
    print("\n" + "=" * 70)
    print("KS TEST RESULTS")
    print("=" * 70)

    features = params['features']
    all_results = []

    for feature in features:
        print(f"\n{feature}:")
        ks_df = stats.run_multiple_ks_tests(df_filtered, feature, group_col, baseline)
        ks_df['feature'] = feature

        for _, row in ks_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row['comparison']}: KS={row['ks_statistic']:.3f}, p={row['p_value']:.2e} {sig}")

        all_results.append(ks_df)

        # Plot distribution
        plot_distributions(df_filtered, feature, group_col, output_dir, TEST_NAME)

    # Combine results
    ks_results_all = pd.concat(all_results, ignore_index=True)
    output.save_csv(ks_results_all, output_dir, TEST_NAME, 'results')

    # Summary plot
    if len(ks_results_all) > 0:
        plot_ks_summary(ks_results_all, output_dir, TEST_NAME)

    # Generate report
    report = generate_report(ks_results_all, baseline, params)
    output.save_report(report, output_dir, TEST_NAME)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'ks_results': ks_results_all,
        'output_dir': output_dir,
    }


def generate_report(ks_results, baseline, params):
    """Generate text report."""
    lines = [
        "=" * 70,
        "KOLMOGOROV-SMIRNOV TEST REPORT",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Data file: {params['data_file']}",
        f"Group column: {params['group_col']}",
        f"Baseline group: {baseline}",
        f"Features analyzed: {len(params['features'])}",
        "",
        "INTERPRETATION",
        "-" * 50,
        "The KS test measures the maximum difference between two cumulative",
        "distribution functions. A significant result (p < 0.05) indicates",
        "the distributions are different.",
        "",
        "RESULTS SUMMARY",
        "-" * 50,
    ]

    sig_results = ks_results[ks_results['significant']]
    lines.append(f"\nSignificant differences: {len(sig_results)}/{len(ks_results)}")

    if len(sig_results) > 0:
        lines.append("\nMost different distributions:")
        top_diffs = sig_results.nlargest(5, 'ks_statistic')
        for _, row in top_diffs.iterrows():
            lines.append(f"  - {row['feature']} ({row['comparison']}): KS={row['ks_statistic']:.3f}")

    lines.extend([
        "",
        "FULL RESULTS",
        "-" * 50,
        ks_results.to_string(),
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
