#!/usr/bin/env python3
"""
Full Outlier Detection Orchestrator
====================================

Complete pipeline that chains: EFA → RF on Factors → Outlier Detection → Recommendations

Mirrors: archive/sabel_price_tier_outliers.py

This orchestrator focuses on a single supplier to identify purchasers who may be
assigned to the wrong price tier based on their order characteristics.

Methodology:
1. EFA - Extract interpretable latent factors from order data
2. Random Forest - Predict expected tier based on order patterns
3. Outlier Detection - Flag purchasers where assigned ≠ predicted tier
4. Generate severity-based recommendations (HIGH/MEDIUM/LOW)

Outputs: outputs/{DATE}-outlier-detection/
    - confusion-matrix.png, factors-by-tier.png, outlier-scatter.png, winrate-by-match.png
    - all-purchasers.csv, outliers.csv, recommendations.csv, factor-importance.csv
    - summary.txt
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
from sklearn.metrics import confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_NAME = 'outlier-detection'

PARAMS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': ['Sabel Steel'],  # Single supplier focus
    'start_date': config.DEFAULT_START_DATE,
    'features': config.DEFAULT_EFA_FEATURES,
    'n_factors': None,  # Auto-detect via Kaiser
    'top_n_tiers': config.DEFAULT_N_TOP_TIERS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')
    ax.set_title('Actual vs Predicted Price Tier')

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'confusion-matrix')


def plot_factors_by_tier(df, factor_cols, output_dir):
    """Create boxplots showing factor distributions by tier."""
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
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'factors-by-tier')


def plot_outlier_scatter(df, factor_cols, outliers, output_dir):
    """Create scatter plot in factor space with outliers highlighted."""
    if len(factor_cols) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get purchaser-level data
    purchaser_factors = df.groupby('PURCHASERCOMPANYNAME')[factor_cols].mean()
    purchaser_tiers = df.groupby('PURCHASERCOMPANYNAME')['TIER'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )

    # Plot all purchasers
    scatter = ax.scatter(
        purchaser_factors[factor_cols[0]],
        purchaser_factors[factor_cols[1]],
        c=pd.Categorical(purchaser_tiers).codes,
        cmap='tab10',
        s=100,
        alpha=0.6
    )

    # Highlight outliers
    if len(outliers) > 0:
        outlier_names = outliers.index.tolist()
        outlier_mask = purchaser_factors.index.isin(outlier_names)
        outlier_factors = purchaser_factors[outlier_mask]

        ax.scatter(
            outlier_factors[factor_cols[0]],
            outlier_factors[factor_cols[1]],
            c='red',
            s=200,
            marker='x',
            linewidths=3,
            label='Outliers'
        )

        # Label outliers
        for name in outlier_names:
            if name in purchaser_factors.index:
                ax.annotate(
                    name.split('.')[0][:15],  # Shorten long names
                    (purchaser_factors.loc[name, factor_cols[0]],
                     purchaser_factors.loc[name, factor_cols[1]]),
                    fontsize=8,
                    alpha=0.8
                )

    ax.set_xlabel(factor_cols[0])
    ax.set_ylabel(factor_cols[1])
    ax.set_title('Purchasers in Factor Space\n(Red X = Potential Outliers)')
    ax.legend()
    plt.colorbar(scatter, label='Tier')

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'outlier-scatter')


def plot_winrate_by_match(df, output_dir):
    """Create win rate comparison for matching vs non-matching tier predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    match_stats = df.groupby(['TIER', 'MATCH'])['ISWON'].mean().unstack()
    match_stats.plot(kind='bar', ax=ax)

    ax.set_xlabel('Assigned Tier')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate: Matching vs Non-Matching Tier Predictions')
    ax.legend(['Mismatch (outlier)', 'Match'])
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'winrate-by-match')


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def identify_outliers(df, factor_cols):
    """
    Identify purchasers whose order patterns don't match their tier.

    Returns:
        purchaser_stats: Full purchaser statistics
        outliers: Subset of purchasers with tier mismatch
    """
    # Aggregate by purchaser
    agg_dict = {
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
        'MARGINSYSTEMRECOMMENDED': 'mean',
    }

    # Add factor columns
    for fc in factor_cols:
        agg_dict[fc] = 'mean'

    purchaser_stats = df.groupby('PURCHASERCOMPANYNAME').agg(agg_dict).round(3)

    # Flatten column names
    purchaser_stats.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate', 'avg_margin'
    ] + [f'avg_{fc}' for fc in factor_cols]

    purchaser_stats = purchaser_stats.sort_values('n_quotes', ascending=False)
    purchaser_stats['tier_match'] = purchaser_stats['assigned_tier'] == purchaser_stats['predicted_tier']

    # Identify outliers (tier mismatch)
    outliers = purchaser_stats[~purchaser_stats['tier_match']].copy()

    return purchaser_stats, outliers


def generate_recommendations(outliers, factor_cols):
    """Generate severity-based recommendations for outliers."""
    recommendations = []

    for idx, row in outliers.iterrows():
        rec = {
            'purchaser': idx,
            'assigned_tier': row['assigned_tier'],
            'predicted_tier': row['predicted_tier'],
            'n_quotes': int(row['n_quotes']),
            'win_rate': row['win_rate'],
            'avg_estimate': row['avg_estimate'],
            'severity': 'LOW',
        }

        # Determine severity and recommendation
        if row['win_rate'] == 0:
            rec['severity'] = 'HIGH'
            rec['recommendation'] = (
                f"URGENT: Review pricing. 0% win rate suggests significant overpricing. "
                f"Consider moving to {row['predicted_tier']}."
            )
        elif row['win_rate'] < 0.15:
            rec['severity'] = 'HIGH'
            rec['recommendation'] = (
                f"Review pricing. Low win rate ({row['win_rate']*100:.0f}%) suggests "
                f"potential overpricing. Consider {row['predicted_tier']}."
            )
        elif row['win_rate'] > 0.80:
            rec['severity'] = 'MEDIUM'
            rec['recommendation'] = (
                f"Revenue opportunity. High win rate ({row['win_rate']*100:.0f}%) may "
                f"indicate underpricing. Consider moving from {row['assigned_tier']} "
                f"toward {row['predicted_tier']}."
            )
        else:
            rec['severity'] = 'LOW'
            rec['recommendation'] = (
                f"Factor profile suggests {row['predicted_tier']} may be a better fit "
                f"than current {row['assigned_tier']}."
            )

        # Add factor profile
        for fc in factor_cols:
            rec[fc] = row[f'avg_{fc}']

        recommendations.append(rec)

    recommendations_df = pd.DataFrame(recommendations)

    # Sort by severity
    if len(recommendations_df) > 0:
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations_df['_sort'] = recommendations_df['severity'].map(severity_order)
        recommendations_df = recommendations_df.sort_values(['_sort', 'n_quotes'], ascending=[True, False])
        recommendations_df = recommendations_df.drop('_sort', axis=1)

    return recommendations_df


def generate_summary_report(accuracy, purchaser_stats, outliers, recommendations, factor_cols, params):
    """Generate comprehensive text summary."""
    lines = [
        "=" * 70,
        "PRICE TIER OUTLIER ANALYSIS",
        "=" * 70,
        "",
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Supplier: {', '.join(params['suppliers'])}",
        f"Model Accuracy: {accuracy*100:.1f}%",
        f"Total Purchasers: {len(purchaser_stats)}",
        f"Outliers Identified: {len(outliers)}",
        "",
    ]

    # High severity
    lines.append("-" * 70)
    lines.append("HIGH SEVERITY OUTLIERS (Requires Immediate Review)")
    lines.append("-" * 70)

    high_sev = recommendations[recommendations['severity'] == 'HIGH'] if len(recommendations) > 0 else pd.DataFrame()
    if len(high_sev) > 0:
        for _, row in high_sev.iterrows():
            lines.append(f"\n{row['purchaser']}")
            lines.append(f"  {row['assigned_tier']} -> {row['predicted_tier']}")
            lines.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
            lines.append(f"  Recommendation: {row['recommendation']}")
    else:
        lines.append("  None identified.")

    # Medium severity
    lines.append("")
    lines.append("-" * 70)
    lines.append("MEDIUM SEVERITY OUTLIERS (Revenue Opportunities)")
    lines.append("-" * 70)

    med_sev = recommendations[recommendations['severity'] == 'MEDIUM'] if len(recommendations) > 0 else pd.DataFrame()
    if len(med_sev) > 0:
        for _, row in med_sev.iterrows():
            lines.append(f"\n{row['purchaser']}")
            lines.append(f"  {row['assigned_tier']} -> {row['predicted_tier']}")
            lines.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
            lines.append(f"  Recommendation: {row['recommendation']}")
    else:
        lines.append("  None identified.")

    # Low severity
    lines.append("")
    lines.append("-" * 70)
    lines.append("LOW SEVERITY OUTLIERS (Factor Profile Mismatches)")
    lines.append("-" * 70)

    low_sev = recommendations[recommendations['severity'] == 'LOW'] if len(recommendations) > 0 else pd.DataFrame()
    if len(low_sev) > 0:
        for _, row in low_sev.iterrows():
            lines.append(f"\n{row['purchaser']}")
            lines.append(f"  {row['assigned_tier']} -> {row['predicted_tier']}")
            lines.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
    else:
        lines.append("  None identified.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(params: dict = None) -> dict:
    """
    Run the full outlier detection pipeline.

    Parameters:
        params: Optional parameter overrides

    Returns:
        Dictionary with all analysis results
    """
    if params is None:
        params = PARAMS

    print("=" * 70)
    print("FULL OUTLIER DETECTION PIPELINE")
    print("=" * 70)
    print(f"Supplier: {', '.join(params['suppliers'])}")

    # Setup output directory
    output_dir = output.get_output_dir(PIPELINE_NAME, params['output_base'])

    # -------------------------------------------------------------------------
    # STEP 1: Load and filter data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    df = data.load_csv(params['data_file'])

    if params['suppliers']:
        df = data.filter_by_suppliers(df, params['suppliers'])

    if params['start_date']:
        df = data.filter_by_date(df, params['start_date'])

    df, tier_map = data.create_tier_labels(df)

    print(f"\nPrice tier distribution:")
    for tier, count in df['TIER'].value_counts().head(params['top_n_tiers'] + 2).items():
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} records ({pct:.1f}%)")

    print(f"\nUnique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")
    print(f"Overall win rate: {df['ISWON'].mean()*100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 2: Run EFA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY FACTOR ANALYSIS")
    print("=" * 70)

    features = params['features']
    scaled_array, scaled_df, valid_indices, scaler = data.standardize_features(df, features)

    # Check factorability
    factorability = efa.check_factorability(scaled_array, features)

    # Determine factors
    eigenvalues, suggested_factors = efa.determine_num_factors(scaled_array, features)
    n_factors = params['n_factors'] or suggested_factors

    print(f"\nEigenvalues > 1: {n_factors} factors")

    # Run EFA
    efa_results = efa.run_efa(scaled_array, features, n_factors)

    # Save loadings
    loadings = efa_results['loadings']
    print(f"\nFactor loadings (Varimax rotation):")
    print(loadings.round(3).to_string())

    output.save_csv(loadings, output_dir, PIPELINE_NAME, 'factor-loadings', index=True)

    # Calculate factor scores
    factor_scores = efa.calculate_factor_scores(
        efa_results['factor_analyzer'], scaled_array, valid_indices
    )
    factor_cols = list(factor_scores.columns)

    print(f"\nTotal variance explained: {efa_results['variance'].loc['Cumulative_Var'].iloc[-1]*100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 3: Train tier predictor
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: TIER PREDICTION MODEL")
    print("=" * 70)

    # Filter to top tiers
    top_tiers = df['TIER'].value_counts().head(params['top_n_tiers']).index.tolist()
    df_top = df.loc[factor_scores.index].copy()
    df_top = df_top[df_top['TIER'].isin(top_tiers)].copy()

    # Add factor scores
    for col in factor_cols:
        df_top[col] = factor_scores.loc[df_top.index, col]

    coverage = len(df_top) / len(df.loc[factor_scores.index]) * 100
    print(f"Using top {params['top_n_tiers']} tiers: {len(df_top):,} records ({coverage:.0f}%)")

    # Train RF on factor scores
    X = df_top[factor_cols]
    y = df_top['TIER']

    rf_results = models.train_rf_classifier(X, y)
    df_top['PREDICTED_TIER'] = rf_results['predictions']
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    accuracy = df_top['MATCH'].mean()
    print(f"\nCross-validation accuracy: {rf_results['cv_scores'].mean()*100:.1f}% (+/- {rf_results['cv_scores'].std()*100:.1f}%)")
    print(f"Overall accuracy: {accuracy*100:.1f}%")

    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in sorted(df_top['TIER'].unique()):
        tier_data = df_top[df_top['TIER'] == tier]
        tier_acc = tier_data['MATCH'].mean()
        print(f"  {tier}: {tier_acc*100:.0f}% ({len(tier_data)} records)")

    # Feature importance
    importance = models.get_feature_importance(rf_results['model'], factor_cols)
    output.save_csv(importance, output_dir, PIPELINE_NAME, 'factor-importance')

    print("\nFactor importance for tier prediction:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"  {row['feature']:25s}: {row['importance']:.3f} {bar}")

    # -------------------------------------------------------------------------
    # STEP 4: Identify outliers
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: OUTLIER IDENTIFICATION")
    print("=" * 70)

    purchaser_stats, outliers = identify_outliers(df_top, factor_cols)

    print(f"\nTotal purchasers analyzed: {len(purchaser_stats)}")
    print(f"POTENTIAL OUTLIERS (tier mismatch): {len(outliers)}")

    # Print outlier details
    if len(outliers) > 0:
        print("\n" + "-" * 50)
        print("OUTLIER DETAILS:")
        print("-" * 50)

        for idx, row in outliers.head(10).iterrows():
            print(f"\n{idx}")
            print(f"   Assigned: {row['assigned_tier']} -> Model predicts: {row['predicted_tier']}")
            print(f"   Quotes: {int(row['n_quotes'])}, Win rate: {row['win_rate']*100:.0f}%")
            print(f"   Avg estimate: ${row['avg_estimate']:,.0f}")

    output.save_csv(purchaser_stats, output_dir, PIPELINE_NAME, 'all-purchasers', index=True)
    output.save_csv(outliers, output_dir, PIPELINE_NAME, 'outliers', index=True)

    # -------------------------------------------------------------------------
    # STEP 5: Generate recommendations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING RECOMMENDATIONS")
    print("=" * 70)

    recommendations = generate_recommendations(outliers, factor_cols)
    output.save_csv(recommendations, output_dir, PIPELINE_NAME, 'recommendations')

    # Count by severity
    if len(recommendations) > 0:
        print("\nRecommendations by severity:")
        for sev in ['HIGH', 'MEDIUM', 'LOW']:
            count = len(recommendations[recommendations['severity'] == sev])
            print(f"  {sev}: {count}")

    # -------------------------------------------------------------------------
    # STEP 6: Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("=" * 70)

    # Confusion matrix
    tier_order = sorted(df_top['TIER'].unique())
    plot_confusion_matrix(df_top['TIER'], df_top['PREDICTED_TIER'], tier_order, output_dir)

    # Factors by tier
    plot_factors_by_tier(df_top, factor_cols, output_dir)

    # Outlier scatter
    plot_outlier_scatter(df_top, factor_cols, outliers, output_dir)

    # Win rate by match
    plot_winrate_by_match(df_top, output_dir)

    # -------------------------------------------------------------------------
    # STEP 7: Generate summary report
    # -------------------------------------------------------------------------
    summary = generate_summary_report(
        accuracy, purchaser_stats, outliers, recommendations, factor_cols, params
    )

    with open(f"{output_dir}/{PIPELINE_NAME}-summary.txt", 'w') as f:
        f.write(summary)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    print(f"\nKey findings:")
    print(f"  - Model accuracy: {accuracy*100:.1f}%")
    print(f"  - Outliers identified: {len(outliers)} purchasers")

    if len(recommendations) > 0:
        high_sev = len(recommendations[recommendations['severity'] == 'HIGH'])
        if high_sev > 0:
            print(f"  - HIGH severity (review urgently): {high_sev}")

    return {
        'df_top': df_top,
        'factor_scores': factor_scores,
        'purchaser_stats': purchaser_stats,
        'outliers': outliers,
        'recommendations': recommendations,
        'importance': importance,
        'output_dir': output_dir,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results = run_pipeline()
