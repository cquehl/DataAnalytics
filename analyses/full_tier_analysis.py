#!/usr/bin/env python3
"""
Full Tier Analysis Orchestrator
================================

Complete pipeline that chains: EFA → ANOVA → Tier Profiles → Interpretation

Mirrors: archive/efa_price_tier.py

This orchestrator uses the modular efa_core library to replicate the
original end-to-end workflow in a more maintainable structure.

Workflow:
1. Load & filter data (Sabel + Coosa, Aug 25+)
2. Standardize features
3. Check factorability
4. Run EFA (auto factors via Kaiser)
5. Calculate factor scores
6. ANOVA: Do factors differ by tier?
7. Calculate tier profiles (mean factor scores + win rates)
8. Generate interpretation report

Outputs: outputs/{DATE}-tier-analysis/
    - scree.png, loadings.png, factors-by-tier.png
    - factorability.csv, loadings.csv, anova.csv, tier-profiles.csv
    - interpretation.txt
"""

import sys
from pathlib import Path

# Hybrid import: works both when pip-installed and when run directly
try:
    from efa_core import data, efa, stats, viz, output, config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from efa_core import data, efa, stats, viz, output, config

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_NAME = 'tier-analysis'

PARAMS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': config.DEFAULT_SUPPLIERS,
    'start_date': config.DEFAULT_START_DATE,
    'features': config.DEFAULT_EFA_FEATURES,
    'n_factors': None,  # Auto-detect via Kaiser
    'top_n_tiers': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_scree(eigenvalues, n_factors, output_dir):
    """Create scree plot showing eigenvalues with Kaiser criterion."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
    ax.set_xlabel('Factor Number', fontsize=11)
    ax.set_ylabel('Eigenvalue', fontsize=11)
    ax.set_title('Scree Plot - Finding the "Elbow"', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(eigenvalues) + 1))

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'scree')


def plot_loadings_heatmap(loadings, output_dir):
    """Create factor loadings heatmap with color-coded strength."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Factor Loadings (Varimax Rotation)', fontsize=12)

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'loadings')


def plot_factors_by_tier(analysis_df, factor_cols, output_dir):
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
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'factors-by-tier')


# =============================================================================
# INTERPRETATION GENERATOR
# =============================================================================
def generate_interpretation(loadings, anova_df, tier_means, factor_cols, params):
    """Generate comprehensive human-readable interpretation of results."""
    lines = []
    lines.append("=" * 70)
    lines.append("EFA PRICE TIER ANALYSIS - INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")

    # Configuration summary
    lines.append("ANALYSIS CONFIGURATION")
    lines.append("-" * 50)
    lines.append(f"Data file: {params['data_file']}")
    lines.append(f"Suppliers: {', '.join(params['suppliers'])}")
    lines.append(f"Start date: {params['start_date']}")
    lines.append(f"Features analyzed: {len(params['features'])}")
    lines.append("")

    # Factor descriptions
    lines.append("WHAT ARE THE FACTORS?")
    lines.append("-" * 50)

    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > config.LOADING_THRESHOLD][col].sort_values(ascending=False)
        if len(high_loaders) > 0:
            lines.append(f"\n{col}:")
            for var, loading in high_loaders.items():
                sign = "higher" if loading > 0 else "lower"
                lines.append(f"  - {sign} {var}")

    lines.append("")
    lines.append("")

    # Which factors discriminate tiers?
    lines.append("WHICH FACTORS DISCRIMINATE BETWEEN TIERS?")
    lines.append("-" * 50)

    sig_factors = anova_df[anova_df['significant']]
    if len(sig_factors) > 0:
        for _, row in sig_factors.iterrows():
            lines.append(f"\n{row['variable']}: YES (p={row['p_value']:.2e}, eta^2={row['eta_squared']:.3f})")
            lines.append(f"  Effect size: {row['effect_size']}")

    non_sig = anova_df[~anova_df['significant']]
    if len(non_sig) > 0:
        lines.append(f"\nNot significant: {', '.join(non_sig['variable'].tolist())}")

    lines.append("")
    lines.append("")

    # Tier profiles
    lines.append("TIER PROFILES (what makes each tier different)")
    lines.append("-" * 50)

    for tier in tier_means.index:
        row = tier_means.loc[tier]
        n_quotes = int(row['n_quotes']) if 'n_quotes' in row else 'N/A'
        win_rate = row['win_rate'] * 100 if 'win_rate' in row else 'N/A'
        lines.append(f"\n{tier} ({n_quotes} quotes, {win_rate:.0f}% win rate):")

        for factor in factor_cols:
            if factor in row:
                val = row[factor]
                if abs(val) > 0.3:
                    direction = "HIGH" if val > 0 else "LOW"
                    lines.append(f"  - {factor}: {direction} ({val:+.2f})")

    lines.append("")
    lines.append("")
    lines.append("=" * 70)
    lines.append("HOW TO USE THESE FINDINGS")
    lines.append("=" * 70)
    lines.append("""
1. FACTOR SCORES AS FEATURES
   The factor scores can be used as inputs to the Random Forest model.
   This gives you INTERPRETABLE features instead of raw variables.

2. TIER PROFILING
   Use the tier profiles to understand what "type" of customer each tier
   represents. This helps explain tier assignments to stakeholders.

3. ANOMALY DETECTION
   If a quote's factor scores don't match its tier's typical profile,
   it may be misassigned.

4. HYBRID APPROACH
   Combine EFA insights (interpretability) with Random Forest (accuracy)
   for a robust validation methodology.
""")

    return "\n".join(lines)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(params: dict = None) -> dict:
    """
    Run the full tier analysis pipeline.

    Parameters:
        params: Optional parameter overrides

    Returns:
        Dictionary with all analysis results
    """
    if params is None:
        params = PARAMS

    print("=" * 70)
    print("FULL TIER ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Suppliers: {', '.join(params['suppliers'])}")
    print(f"Start date: {params['start_date']}")

    # Setup output directory
    output_dir = output.get_output_dir(PIPELINE_NAME, params['output_base'])

    # -------------------------------------------------------------------------
    # STEP 1: Load and filter data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND FILTERING DATA")
    print("=" * 70)

    df = data.load_csv(params['data_file'])

    if params['suppliers']:
        df = data.filter_by_suppliers(df, params['suppliers'])

    if params['start_date']:
        df = data.filter_by_date(df, params['start_date'])

    df, tier_map = data.create_tier_labels(df)

    print(f"\nUnique tiers: {df['TIER'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")

    # -------------------------------------------------------------------------
    # STEP 2: Standardize features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: STANDARDIZING FEATURES")
    print("=" * 70)

    features = params['features']
    print(f"Features: {', '.join(features)}")

    scaled_array, scaled_df, valid_indices, scaler = data.standardize_features(df, features)

    # -------------------------------------------------------------------------
    # STEP 3: Check factorability
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: CHECKING FACTORABILITY")
    print("=" * 70)

    factorability = efa.check_factorability(scaled_array, features)
    factorability_df = efa.get_factorability_summary(factorability)
    output.save_csv(factorability_df, output_dir, PIPELINE_NAME, 'factorability')

    # -------------------------------------------------------------------------
    # STEP 4: Determine number of factors
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: DETERMINING NUMBER OF FACTORS")
    print("=" * 70)

    eigenvalues, suggested_factors = efa.determine_num_factors(scaled_array, features)
    n_factors = params['n_factors'] or suggested_factors

    print(f"\nKaiser criterion suggests: {suggested_factors} factors")
    print(f"Using: {n_factors} factors")

    plot_scree(eigenvalues, n_factors, output_dir)

    # -------------------------------------------------------------------------
    # STEP 5: Run EFA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 5: FACTOR ANALYSIS ({n_factors} factors, Varimax rotation)")
    print("=" * 70)

    efa_results = efa.run_efa(scaled_array, features, n_factors)

    # Save and display loadings
    loadings = efa_results['loadings']
    output.save_csv(loadings, output_dir, PIPELINE_NAME, 'loadings', index=True)
    plot_loadings_heatmap(loadings, output_dir)

    print("\nFactor Loadings:")
    print(loadings.round(3).to_string())

    print(f"\nTotal variance explained: {efa_results['variance'].loc['Cumulative_Var'].iloc[-1]*100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 6: Calculate factor scores
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: CALCULATING FACTOR SCORES")
    print("=" * 70)

    factor_scores = efa.calculate_factor_scores(
        efa_results['factor_analyzer'], scaled_array, valid_indices
    )
    factor_cols = list(factor_scores.columns)

    print(f"Calculated factor scores for {len(factor_scores):,} quotes")
    output.save_csv(factor_scores, output_dir, PIPELINE_NAME, 'factor-scores', index=True)

    # -------------------------------------------------------------------------
    # STEP 7: Analyze factors by tier (ANOVA)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: ANALYZING FACTORS BY PRICE TIER (ANOVA)")
    print("=" * 70)

    # Merge factor scores with tier info
    analysis_df = df.loc[factor_scores.index, ['TIER', 'ISWON', 'PURCHASERCOMPANYNAME']].copy()
    analysis_df = analysis_df.join(factor_scores)

    # Filter to top tiers
    top_tiers = df['TIER'].value_counts().head(params['top_n_tiers']).index.tolist()
    analysis_df = analysis_df[analysis_df['TIER'].isin(top_tiers)]

    print(f"\nAnalyzing top {params['top_n_tiers']} tiers ({len(analysis_df):,} records)")

    # Run ANOVA
    anova_df = stats.run_anova_multi(analysis_df, factor_cols, 'TIER')
    stats.print_anova_results(anova_df)
    output.save_csv(anova_df, output_dir, PIPELINE_NAME, 'anova')

    # -------------------------------------------------------------------------
    # STEP 8: Calculate tier profiles
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: TIER PROFILES")
    print("=" * 70)

    tier_means = stats.calculate_group_means(analysis_df, factor_cols, 'TIER')
    tier_means['n_quotes'] = analysis_df.groupby('TIER').size()
    tier_means['win_rate'] = analysis_df.groupby('TIER')['ISWON'].mean().round(3)

    print("\nTier Profiles (mean factor scores):")
    print(tier_means.to_string())

    output.save_csv(tier_means, output_dir, PIPELINE_NAME, 'tier-profiles', index=True)

    # Plot factors by tier
    plot_factors_by_tier(analysis_df, factor_cols, output_dir)

    # -------------------------------------------------------------------------
    # STEP 9: Generate interpretation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: GENERATING INTERPRETATION")
    print("=" * 70)

    interpretation = generate_interpretation(loadings, anova_df, tier_means, factor_cols, params)

    with open(f"{output_dir}/{PIPELINE_NAME}-interpretation.txt", 'w') as f:
        f.write(interpretation)

    print(f"Interpretation saved to: {output_dir}/{PIPELINE_NAME}-interpretation.txt")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'df': df,
        'factor_scores': factor_scores,
        'loadings': loadings,
        'anova': anova_df,
        'tier_means': tier_means,
        'interpretation': interpretation,
        'output_dir': output_dir,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results = run_pipeline()
