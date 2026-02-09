#!/usr/bin/env python3
"""
Outlier Detection Test Script
=============================

Combines EFA with Random Forest to identify price tier outliers.
Outliers are purchasers whose order patterns don't match their tier.

Parameters:
    data_file     - Path to input CSV
    suppliers     - List of suppliers to filter
    start_date    - Filter from date
    features      - EFA features
    n_factors     - Number of factors (None = auto)
    n_top_tiers   - Number of top tiers to include

Outputs:
    - Outlier scatter plot (PNG)
    - Confusion matrix (PNG)
    - Win rate by match (PNG)
    - All purchasers analysis (CSV)
    - Outliers only (CSV)
    - Recommendations (CSV)
    - Factor importance (CSV)
    - Summary report (TXT)
"""

import sys
from pathlib import Path

# Hybrid import: works both when pip-installed and when run directly
try:
    from efa_core import data, efa, models, stats, viz, output, config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from efa_core import data, efa, models, stats, viz, output, config

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
    'suppliers': ['Sabel Steel'],  # Focus on single supplier
    'start_date': config.DEFAULT_START_DATE,
    'features': config.DEFAULT_EFA_FEATURES,
    'n_factors': None,  # Auto-detect
    'n_top_tiers': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}

TEST_NAME = 'outlier-detection'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, labels, output_dir, test_name):
    """Create confusion matrix."""
    from sklearn.metrics import confusion_matrix as cm
    conf = cm(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'confusion-matrix')


def plot_outlier_scatter(df, factor_cols, outliers, output_dir, test_name):
    """Create scatter plot highlighting outliers."""
    if len(factor_cols) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    non_outliers = df[~df.index.isin(outliers.index)]
    ax.scatter(non_outliers[factor_cols[0]], non_outliers[factor_cols[1]],
               c='lightgray', alpha=0.5, label='Normal', s=30)

    # Highlight outliers
    outlier_data = df[df.index.isin(outliers.index)]
    scatter = ax.scatter(outlier_data[factor_cols[0]], outlier_data[factor_cols[1]],
                        c='red', alpha=0.8, label='Outlier', s=50, marker='x')

    ax.set_xlabel(factor_cols[0])
    ax.set_ylabel(factor_cols[1])
    ax.set_title('Factor Space with Outliers Highlighted')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'outlier-scatter')


def plot_factors_by_tier(df, factor_cols, output_dir, test_name):
    """Create boxplots of factors by tier."""
    n_factors = len(factor_cols)
    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))

    if n_factors == 1:
        axes = [axes]

    tier_order = df['TIER'].value_counts().index.tolist()

    for ax, factor in zip(axes, factor_cols):
        sns.boxplot(data=df, x='TIER', y=factor, order=tier_order, ax=ax)
        ax.set_title(f'{factor} by Tier')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'factors-by-tier')


def plot_winrate_by_match(purchaser_stats, output_dir, test_name):
    """Create win rate comparison for matching vs non-matching tiers."""
    fig, ax = plt.subplots(figsize=(8, 6))

    match_groups = purchaser_stats.groupby('tier_match').agg({
        'win_rate': 'mean',
        'n_quotes': 'sum'
    })

    colors = [viz.get_colors()['significant'], viz.get_colors()['accent']]
    labels = ['Non-matching', 'Matching']

    if len(match_groups) == 2:
        bars = ax.bar(labels, match_groups['win_rate'], color=colors)

        for bar, (_, row) in zip(bars, match_groups.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={int(row["n_quotes"])}', ha='center', fontsize=10)

    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate: Tier Match vs Non-Match')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'winrate-by-match')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(params: dict) -> dict:
    """Run outlier detection pipeline."""
    print("=" * 70)
    print("OUTLIER DETECTION ANALYSIS")
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

    print(f"\nUnique tiers: {df['TIER'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")

    # Run EFA
    print("\n" + "=" * 70)
    print("RUNNING EFA")
    print("=" * 70)

    features = params['features']
    scaled_array, scaled_df, valid_indices, scaler = data.standardize_features(df, features)

    # Determine factors
    eigenvalues, suggested_factors = efa.determine_num_factors(scaled_array, features)
    n_factors = params['n_factors'] or suggested_factors

    # Run EFA
    efa_results = efa.run_efa(scaled_array, features, n_factors)
    factor_scores = efa.calculate_factor_scores(
        efa_results['factor_analyzer'], scaled_array, valid_indices
    )
    factor_cols = list(factor_scores.columns)

    # Save loadings
    output.save_csv(efa_results['loadings'], output_dir, TEST_NAME, 'factor-loadings', index=True)

    # Train tier predictor
    print("\n" + "=" * 70)
    print("TRAINING TIER PREDICTOR")
    print("=" * 70)

    # Filter to top tiers
    top_tiers = df['TIER'].value_counts().head(params['n_top_tiers']).index.tolist()
    df_top = df.loc[factor_scores.index].copy()
    df_top = df_top[df_top['TIER'].isin(top_tiers)].copy()

    # Add factor scores
    for col in factor_cols:
        df_top[col] = factor_scores.loc[df_top.index, col]

    print(f"\nTop {params['n_top_tiers']} tiers: {len(df_top):,} records")

    # Train RF on factor scores
    X = df_top[factor_cols]
    y = df_top['TIER']

    rf_results = models.train_rf_classifier(X, y)
    df_top['PREDICTED_TIER'] = rf_results['predictions']
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    # Feature importance
    importance = models.get_feature_importance(rf_results['model'], factor_cols)
    output.save_csv(importance, output_dir, TEST_NAME, 'factor-importance')

    # Confusion matrix
    plot_confusion_matrix(
        y.values, rf_results['predictions'],
        sorted(df_top['TIER'].unique()),
        output_dir, TEST_NAME
    )

    # Factors by tier
    plot_factors_by_tier(df_top, factor_cols, output_dir, TEST_NAME)

    # Identify outliers
    print("\n" + "=" * 70)
    print("IDENTIFYING OUTLIERS")
    print("=" * 70)

    purchaser_stats, outliers = identify_outliers(df_top, factor_cols)

    output.save_csv(purchaser_stats, output_dir, TEST_NAME, 'all-purchasers', index=True)
    output.save_csv(outliers, output_dir, TEST_NAME, 'outliers', index=True)

    print(f"\nTotal purchasers: {len(purchaser_stats)}")
    print(f"Outliers identified: {len(outliers)}")

    if len(outliers) > 0:
        print("\nTop outliers:")
        print(outliers.head(10).to_string())

    # Plot outliers
    plot_outlier_scatter(df_top, factor_cols, outliers, output_dir, TEST_NAME)
    plot_winrate_by_match(purchaser_stats, output_dir, TEST_NAME)

    # Generate recommendations
    recommendations = generate_recommendations(purchaser_stats, outliers)
    output.save_csv(recommendations, output_dir, TEST_NAME, 'recommendations')

    # Generate report
    report = generate_report(
        rf_results, importance, purchaser_stats, outliers, recommendations, params
    )
    output.save_report(report, output_dir, TEST_NAME, 'summary')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'df_top': df_top,
        'factor_scores': factor_scores,
        'purchaser_stats': purchaser_stats,
        'outliers': outliers,
        'recommendations': recommendations,
        'output_dir': output_dir,
    }


def identify_outliers(df, factor_cols):
    """Identify purchasers whose tier doesn't match their order profile."""
    purchaser_stats = df.groupby('PURCHASERCOMPANYNAME').agg({
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': 'mean',
        'ESTIMATEKEY': 'count',
        **{col: 'mean' for col in factor_cols}
    }).round(3)

    purchaser_stats.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate', 'win_rate', 'n_quotes'
    ] + [f'avg_{col}' for col in factor_cols]

    purchaser_stats['tier_match'] = purchaser_stats['assigned_tier'] == purchaser_stats['predicted_tier']

    # Outliers: low match rate and tier mismatch
    outliers = purchaser_stats[
        (~purchaser_stats['tier_match']) &
        (purchaser_stats['match_rate'] < 0.5)
    ].sort_values('match_rate')

    return purchaser_stats, outliers


def generate_recommendations(purchaser_stats, outliers):
    """Generate actionable recommendations."""
    recommendations = []

    for idx, row in outliers.iterrows():
        if row['win_rate'] < 0.1 and row['n_quotes'] >= 3:
            severity = 'HIGH'
            action = f"Consider moving to {row['predicted_tier']}"
        elif row['win_rate'] < 0.2:
            severity = 'MEDIUM'
            action = f"Review tier assignment, model suggests {row['predicted_tier']}"
        else:
            severity = 'LOW'
            action = "Monitor - tier mismatch but acceptable win rate"

        recommendations.append({
            'purchaser': idx,
            'severity': severity,
            'current_tier': row['assigned_tier'],
            'suggested_tier': row['predicted_tier'],
            'win_rate': row['win_rate'],
            'n_quotes': row['n_quotes'],
            'match_rate': row['match_rate'],
            'action': action,
        })

    return pd.DataFrame(recommendations)


def generate_report(rf_results, importance, purchaser_stats, outliers, recommendations, params):
    """Generate summary report."""
    lines = [
        "=" * 70,
        "OUTLIER DETECTION SUMMARY",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Suppliers: {params['suppliers']}",
        f"Data file: {params['data_file']}",
        f"Top tiers: {params['n_top_tiers']}",
        "",
        "MODEL PERFORMANCE",
        "-" * 50,
        f"Accuracy: {rf_results['accuracy']*100:.1f}%",
        "",
        "FACTOR IMPORTANCE",
        "-" * 50,
    ]

    for _, row in importance.iterrows():
        lines.append(f"  {row['feature']}: {row['importance']:.3f}")

    lines.extend([
        "",
        "OUTLIER SUMMARY",
        "-" * 50,
        f"Total purchasers analyzed: {len(purchaser_stats)}",
        f"Outliers identified: {len(outliers)}",
        f"Outlier rate: {len(outliers)/len(purchaser_stats)*100:.1f}%",
    ])

    if len(recommendations) > 0:
        high = recommendations[recommendations['severity'] == 'HIGH']
        medium = recommendations[recommendations['severity'] == 'MEDIUM']
        low = recommendations[recommendations['severity'] == 'LOW']

        lines.extend([
            "",
            f"HIGH severity: {len(high)}",
            f"MEDIUM severity: {len(medium)}",
            f"LOW severity: {len(low)}",
            "",
            "TOP RECOMMENDATIONS",
            "-" * 50,
        ])

        for _, rec in recommendations.head(10).iterrows():
            lines.append(f"\n{rec['purchaser']} [{rec['severity']}]:")
            lines.append(f"  Current: {rec['current_tier']} → Suggested: {rec['suggested_tier']}")
            lines.append(f"  Win rate: {rec['win_rate']*100:.0f}%, Quotes: {int(rec['n_quotes'])}")
            lines.append(f"  Action: {rec['action']}")

    lines.extend(["", "=" * 70])

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    params = {**DEFAULTS}
    results = run_analysis(params)
