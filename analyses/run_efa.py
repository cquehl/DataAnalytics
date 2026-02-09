#!/usr/bin/env python3
"""
EFA Analysis Test Script
========================

Runs Exploratory Factor Analysis on order characteristics to discover
latent factors and analyze their relationship to price tiers.

Parameters:
    data_file    - Path to input CSV
    suppliers    - List of suppliers to filter (None = all)
    start_date   - Filter from date (None = no filter)
    n_factors    - Number of factors (None = auto-detect via Kaiser)
    features     - List of features for EFA
    top_n_tiers  - Number of top tiers to include in tier analysis

Outputs:
    - Scree plot (PNG)
    - Factor loadings heatmap (PNG)
    - Factor boxplots by tier (PNG)
    - Factorability tests (CSV)
    - Factor loadings (CSV)
    - Factor scores (CSV)
    - ANOVA results (CSV)
    - Text report (TXT)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from efa_core import data, efa, stats, viz, output
import config

# =============================================================================
# PARAMETERS
# =============================================================================
DEFAULTS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': config.DEFAULT_SUPPLIERS,
    'start_date': config.DEFAULT_START_DATE,
    'n_factors': None,  # None = auto-detect via Kaiser
    'features': config.DEFAULT_EFA_FEATURES,
    'top_n_tiers': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}

TEST_NAME = 'efa'


# =============================================================================
# PLOTTING FUNCTIONS (self-contained for portability)
# =============================================================================
def plot_scree(eigenvalues, n_factors, output_dir, test_name):
    """Create scree plot showing eigenvalues."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
    ax.set_xlabel('Factor Number')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(eigenvalues) + 1))

    output.save_figure(fig, output_dir, test_name, 'scree')


def plot_loadings_heatmap(loadings, output_dir, test_name):
    """Create factor loadings heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Factor Loadings (Varimax Rotation)')

    output.save_figure(fig, output_dir, test_name, 'loadings')


def plot_correlation_matrix(scaled_df, output_dir, test_name):
    """Create correlation matrix heatmap."""
    corr_matrix = scaled_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')

    output.save_figure(fig, output_dir, test_name, 'correlation')
    return corr_matrix


def plot_factors_by_tier(analysis_df, factor_cols, output_dir, test_name):
    """Create boxplots showing factor distributions by tier."""
    n_factors = len(factor_cols)
    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))

    if n_factors == 1:
        axes = [axes]

    tier_order = analysis_df['TIER'].value_counts().index.tolist()

    for ax, factor in zip(axes, factor_cols):
        sns.boxplot(data=analysis_df, x='TIER', y=factor, order=tier_order, ax=ax)
        ax.set_title(f'{factor} by Tier')
        ax.set_xlabel('Price Tier')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'factors-by-tier')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(params: dict) -> dict:
    """
    Run the EFA analysis pipeline.

    Parameters:
        params: Dictionary with analysis parameters

    Returns:
        Dictionary with all analysis results
    """
    print("=" * 70)
    print("EFA ANALYSIS")
    print("=" * 70)

    # Setup output directory
    output_dir = output.get_output_dir(TEST_NAME, params['output_base'])

    # Step 1: Load and filter data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    df = data.load_csv(params['data_file'])

    if params['suppliers']:
        df = data.filter_by_suppliers(df, params['suppliers'])

    if params['start_date']:
        df = data.filter_by_date(df, params['start_date'])

    df, tier_map = data.create_tier_labels(df)

    print(f"\nUnique tiers: {df['TIER'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")

    # Step 2: Standardize features
    print("\n" + "=" * 70)
    print("STEP 2: STANDARDIZING FEATURES")
    print("=" * 70)

    features = params['features']
    print(f"Features: {', '.join(features)}")

    scaled_array, scaled_df, valid_indices, scaler = data.standardize_features(df, features)

    # Step 3: Correlation matrix
    print("\n" + "=" * 70)
    print("STEP 3: CORRELATION MATRIX")
    print("=" * 70)

    corr_matrix = plot_correlation_matrix(scaled_df, output_dir, TEST_NAME)

    # Step 4: Check factorability
    factorability = efa.check_factorability(scaled_array, features)
    factorability_df = efa.get_factorability_summary(factorability)
    output.save_csv(factorability_df, output_dir, TEST_NAME, 'factorability')

    # Step 5: Determine number of factors
    eigenvalues, suggested_factors = efa.determine_num_factors(scaled_array, features)
    plot_scree(eigenvalues, suggested_factors, output_dir, TEST_NAME)

    # Step 6: Run EFA
    n_factors = params['n_factors'] or suggested_factors
    efa_results = efa.run_efa(scaled_array, features, n_factors)

    # Save loadings
    output.save_csv(efa_results['loadings'], output_dir, TEST_NAME, 'loadings', index=True)
    plot_loadings_heatmap(efa_results['loadings'], output_dir, TEST_NAME)

    # Save communalities
    output.save_csv(efa_results['communalities'], output_dir, TEST_NAME, 'communalities', index=True)

    # Step 7: Calculate factor scores
    factor_scores = efa.calculate_factor_scores(
        efa_results['factor_analyzer'], scaled_array, valid_indices
    )
    output.save_csv(factor_scores, output_dir, TEST_NAME, 'scores', index=True)

    # Step 8: Analyze by tier
    print("\n" + "=" * 70)
    print("STEP 8: ANALYZING FACTORS BY TIER")
    print("=" * 70)

    # Merge factor scores with tier info
    analysis_df = df.loc[factor_scores.index, ['TIER', 'ISWON', 'PURCHASERCOMPANYNAME']].copy()
    analysis_df = analysis_df.join(factor_scores)

    # Filter to top tiers
    top_tiers = df['TIER'].value_counts().head(params['top_n_tiers']).index.tolist()
    analysis_df = analysis_df[analysis_df['TIER'].isin(top_tiers)]

    print(f"\nAnalyzing top {params['top_n_tiers']} tiers ({len(analysis_df):,} records)")

    factor_cols = list(factor_scores.columns)

    # Run ANOVA
    anova_df = stats.run_anova_multi(analysis_df, factor_cols, 'TIER')
    stats.print_anova_results(anova_df)
    output.save_csv(anova_df, output_dir, TEST_NAME, 'anova')

    # Calculate tier means
    tier_means = stats.calculate_group_means(analysis_df, factor_cols, 'TIER')
    tier_means['win_rate'] = analysis_df.groupby('TIER')['ISWON'].mean().round(3)
    output.save_csv(tier_means, output_dir, TEST_NAME, 'tier-profiles', index=True)

    print("\nTier Profiles:")
    print(tier_means.to_string())

    # Plot factors by tier
    plot_factors_by_tier(analysis_df, factor_cols, output_dir, TEST_NAME)

    # Step 9: Generate report
    report = generate_report(
        efa_results, factorability, anova_df, tier_means, factor_cols, params
    )
    output.save_report(report, output_dir, TEST_NAME)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'df': df,
        'factor_scores': factor_scores,
        'loadings': efa_results['loadings'],
        'anova': anova_df,
        'tier_means': tier_means,
        'output_dir': output_dir,
    }


def generate_report(efa_results, factorability, anova_df, tier_means, factor_cols, params):
    """Generate text report summarizing analysis."""
    lines = [
        "=" * 70,
        "EFA PRICE TIER ANALYSIS REPORT",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Data file: {params['data_file']}",
        f"Suppliers: {params['suppliers']}",
        f"Start date: {params['start_date']}",
        f"Features: {len(params['features'])} variables",
        f"Factors extracted: {efa_results['n_factors']}",
        "",
        "FACTORABILITY",
        "-" * 50,
        f"Bartlett's test: p={factorability['bartlett_p_value']:.2e} ({'PASS' if factorability['bartlett_pass'] else 'FAIL'})",
        f"KMO: {factorability['kmo_overall']:.3f} ({factorability['kmo_label']})",
        "",
        "FACTOR LOADINGS",
        "-" * 50,
    ]

    loadings = efa_results['loadings']
    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > config.LOADING_THRESHOLD][col]
        if len(high_loaders) > 0:
            lines.append(f"\n{col}:")
            for var, loading in high_loaders.items():
                sign = "+" if loading > 0 else "-"
                lines.append(f"  {sign} {var}: {loading:.2f}")

    lines.extend([
        "",
        f"Total variance explained: {efa_results['variance'].loc['Cumulative_Var'].iloc[-1]*100:.1f}%",
        "",
        "ANOVA RESULTS (Factor differences by tier)",
        "-" * 50,
    ])

    for _, row in anova_df.iterrows():
        sig = "SIGNIFICANT" if row['significant'] else "not significant"
        lines.append(f"{row['variable']}: F={row['f_statistic']:.2f}, p={row['p_value']:.2e}, eta^2={row['eta_squared']:.3f} ({sig})")

    lines.extend([
        "",
        "TIER PROFILES",
        "-" * 50,
        tier_means.to_string(),
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
