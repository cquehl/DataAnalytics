#!/usr/bin/env python3
"""
Quick Validation Orchestrator
==============================

Simplified pipeline using Random Forest on raw features (no EFA step).

Mirrors: archive/price_tier_validation.py

This is a faster validation approach that:
1. Profiles tiers with aggregate statistics
2. Trains RF on raw numeric features
3. Analyzes purchaser-level predictions
4. Generates actionable recommendations

Best for: Quick assessment of tier assignments without factor interpretation.

Outputs: outputs/{DATE}-quick-validation/
    - tier-profiles.csv, purchaser-analysis.csv, potential-misassignments.csv
    - recommendations.csv, feature-importance.csv, report.txt
"""

import sys
from pathlib import Path

# Hybrid import: works both when pip-installed and when run directly
try:
    from efa_core import data, models, viz, output, config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from efa_core import data, models, viz, output, config

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
PIPELINE_NAME = 'quick-validation'

PARAMS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': config.DEFAULT_SUPPLIERS,
    'start_date': config.DEFAULT_START_DATE,
    'features': [  # Raw numeric features (no EFA)
        'ESTIMATETOTALPRICE',
        'ESTIMATETOTALQUANTITY',
        'QTYPERLINEITEM',
        'LINEITEMUNITPRICE',
        'MATERIALUNITWEIGHT',
        'MATERIALCENTWEIGHTCOST',
    ],
    'n_top_tiers': config.DEFAULT_N_TOP_TIERS,
    'cv_folds': config.RF_CV_FOLDS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_feature_importance(importance_df, output_dir):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color=viz.get_colors()['primary'])

    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for Tier Prediction')

    for bar, imp in zip(bars, importance_df['importance']):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'feature-importance')


def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')
    ax.set_title('Confusion Matrix: Actual vs Predicted Tier')

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'confusion-matrix')


def plot_tier_profiles(profiles_df, output_dir):
    """Create tier profile comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['win_rate', 'avg_estimate', 'avg_unit_price', 'avg_margin']
    titles = ['Win Rate by Tier', 'Avg Estimate Value', 'Avg Unit Price', 'Avg System Margin']

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        if metric in profiles_df.columns:
            values = profiles_df[metric]
            colors = [viz.get_colors()['primary'] if v > values.mean() else viz.get_colors()['accent']
                     for v in values]

            ax.bar(profiles_df['tier'], values, color=colors)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)

            if 'rate' in metric:
                ax.set_ylim(0, 1)
                ax.axhline(y=values.mean(), color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output.save_figure(fig, output_dir, PIPELINE_NAME, 'tier-profiles')


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def profile_tiers(df):
    """Generate comprehensive tier profiles with key metrics."""
    profiles = []

    for tier in df['TIER'].value_counts().index:
        sub = df[df['TIER'] == tier]
        profile = {
            'tier': tier,
            'n_quotes': len(sub),
            'n_purchasers': sub['PURCHASERCOMPANYNAME'].nunique(),
            'win_rate': sub['ISWON'].mean(),
            'avg_estimate': sub['ESTIMATETOTALPRICE'].mean(),
            'avg_quantity': sub['ESTIMATETOTALQUANTITY'].mean(),
            'avg_unit_price': sub['LINEITEMUNITPRICE'].mean(),
            'avg_margin': sub['MARGINSYSTEMRECOMMENDED'].mean() if 'MARGINSYSTEMRECOMMENDED' in sub.columns else None,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


def analyze_purchasers(df):
    """Aggregate predictions at purchaser level."""
    purchaser_analysis = df.groupby('PURCHASERCOMPANYNAME').agg({
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
        'MARGINSYSTEMRECOMMENDED': 'mean' if 'MARGINSYSTEMRECOMMENDED' in df.columns else 'first',
    }).round(3)

    purchaser_analysis.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate', 'avg_margin'
    ]

    return purchaser_analysis.sort_values('n_quotes', ascending=False)


def generate_recommendations(purchaser_analysis, misassignments):
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Low win rate with potential misassignment
    for idx, row in misassignments.iterrows():
        rec = {
            'purchaser': idx,
            'issue': '',
            'current_tier': row['assigned_tier'],
            'suggested_tier': row['predicted_tier'],
            'win_rate': row['win_rate'],
            'n_quotes': int(row['n_quotes']),
            'priority': 'LOW',
        }

        if row['win_rate'] < 0.05:
            rec['issue'] = 'Critical: Near-zero win rate with tier mismatch'
            rec['priority'] = 'HIGH'
        elif row['win_rate'] < 0.15:
            rec['issue'] = 'Low win rate suggests overpricing'
            rec['priority'] = 'HIGH'
        elif row['win_rate'] > 0.80:
            rec['issue'] = 'High win rate may indicate underpricing'
            rec['priority'] = 'MEDIUM'
        else:
            rec['issue'] = 'Tier assignment differs from order pattern'
            rec['priority'] = 'LOW'

        recommendations.append(rec)

    # Zero win rate purchasers (even if tier matches)
    zero_wins = purchaser_analysis[purchaser_analysis['win_rate'] == 0]
    for idx, row in zero_wins.iterrows():
        if idx not in misassignments.index:  # Don't duplicate
            recommendations.append({
                'purchaser': idx,
                'issue': 'Zero win rate - pricing review needed',
                'current_tier': row['assigned_tier'],
                'suggested_tier': 'Review needed',
                'win_rate': 0,
                'n_quotes': int(row['n_quotes']),
                'priority': 'HIGH',
            })

    rec_df = pd.DataFrame(recommendations)

    # Sort by priority
    if len(rec_df) > 0:
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        rec_df['_sort'] = rec_df['priority'].map(priority_order)
        rec_df = rec_df.sort_values(['_sort', 'n_quotes'], ascending=[True, False])
        rec_df = rec_df.drop('_sort', axis=1)

    return rec_df


def generate_report(tier_profiles, purchaser_analysis, misassignments, recommendations, rf_results, importance, params):
    """Generate comprehensive text report."""
    lines = [
        "=" * 80,
        "QUICK PRICE TIER VALIDATION REPORT",
        "=" * 80,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Data file: {params['data_file']}",
        f"Suppliers: {', '.join(params['suppliers'])}",
        f"Date range: {params['start_date']} onwards",
        f"Top tiers analyzed: {params['n_top_tiers']}",
        "",
        "MODEL PERFORMANCE",
        "-" * 50,
        f"Cross-validation accuracy: {rf_results['cv_scores'].mean()*100:.1f}% (+/- {rf_results['cv_scores'].std()*100:.1f}%)",
        f"Overall accuracy: {rf_results['accuracy']*100:.1f}%",
        "",
        "TIER PROFILES",
        "-" * 50,
    ]

    for _, row in tier_profiles.iterrows():
        lines.append(f"\n{row['tier']} ({int(row['n_quotes'])} quotes, {row['win_rate']*100:.0f}% win rate):")
        lines.append(f"  Purchasers: {int(row['n_purchasers'])}")
        lines.append(f"  Avg estimate: ${row['avg_estimate']:,.0f}")
        lines.append(f"  Avg unit price: ${row['avg_unit_price']:.2f}")
        if row['avg_margin'] is not None:
            lines.append(f"  Avg margin: {row['avg_margin']:.1f}%")

    lines.extend([
        "",
        "FEATURE IMPORTANCE",
        "-" * 50,
    ])

    for _, row in importance.iterrows():
        lines.append(f"  {row['feature']}: {row['importance']:.3f}")

    lines.extend([
        "",
        "PURCHASER ANALYSIS",
        "-" * 50,
        f"Total purchasers: {len(purchaser_analysis)}",
        f"Potential misassignments: {len(misassignments)}",
        f"Zero win rate accounts: {len(purchaser_analysis[purchaser_analysis['win_rate'] == 0])}",
    ])

    if len(misassignments) > 0:
        lines.extend([
            "",
            "POTENTIAL MISASSIGNMENTS",
            "-" * 50,
        ])

        for idx, row in misassignments.head(15).iterrows():
            lines.append(f"\n{idx}:")
            lines.append(f"  Current: {row['assigned_tier']} -> Predicted: {row['predicted_tier']}")
            lines.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {int(row['n_quotes'])}")
            lines.append(f"  Avg estimate: ${row['avg_estimate']:,.0f}")

    if len(recommendations) > 0:
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 50,
        ])

        high_priority = recommendations[recommendations['priority'] == 'HIGH']
        if len(high_priority) > 0:
            lines.append("\nHIGH PRIORITY:")
            for _, row in high_priority.head(10).iterrows():
                lines.append(f"  - {row['purchaser']}: {row['issue']}")
                lines.append(f"    {row['current_tier']} -> {row['suggested_tier']}")

        medium_priority = recommendations[recommendations['priority'] == 'MEDIUM']
        if len(medium_priority) > 0:
            lines.append("\nMEDIUM PRIORITY:")
            for _, row in medium_priority.head(5).iterrows():
                lines.append(f"  - {row['purchaser']}: {row['issue']}")

    lines.extend([
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(params: dict = None) -> dict:
    """
    Run the quick validation pipeline.

    Parameters:
        params: Optional parameter overrides

    Returns:
        Dictionary with all analysis results
    """
    if params is None:
        params = PARAMS

    print("=" * 80)
    print("QUICK PRICE TIER VALIDATION")
    print("=" * 80)
    print(f"\nData file: {params['data_file']}")
    print(f"Suppliers: {', '.join(params['suppliers'])}")
    print(f"Date range: {params['start_date']} onwards")

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

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique tiers: {df['PRICETIERKEY'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")
    print(f"Overall win rate: {df['ISWON'].mean()*100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 2: Profile tiers
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: TIER PROFILES")
    print("=" * 70)

    tier_profiles = profile_tiers(df)

    for _, row in tier_profiles.head(params['n_top_tiers']).iterrows():
        print(f"\n{row['tier']} ({int(row['n_quotes'])} quotes, {row['win_rate']*100:.0f}% win rate):")
        print(f"  Purchasers: {int(row['n_purchasers'])}")
        print(f"  Avg estimate: ${row['avg_estimate']:,.0f}")
        print(f"  Avg quantity: {row['avg_quantity']:.1f}")
        print(f"  Avg unit price: ${row['avg_unit_price']:.2f}")
        if row['avg_margin'] is not None:
            print(f"  Avg margin: {row['avg_margin']:.1f}%")

    output.save_csv(tier_profiles, output_dir, PIPELINE_NAME, 'tier-profiles')
    plot_tier_profiles(tier_profiles.head(params['n_top_tiers']), output_dir)

    # -------------------------------------------------------------------------
    # STEP 3: Train model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: MODEL TRAINING (Top {params['n_top_tiers']} tiers)")
    print("=" * 70)

    # Filter to top tiers
    tier_counts = df['PRICETIERKEY'].value_counts()
    top_tiers = tier_counts.head(params['n_top_tiers']).index.tolist()
    df_top = df[df['PRICETIERKEY'].isin(top_tiers)].copy()

    coverage = len(df_top) / len(df) * 100
    print(f"\nUsing {len(df_top):,} records ({coverage:.0f}% of data)")

    # Prepare features
    features = params['features']
    X = df_top[features].fillna(0)
    y = df_top['TIER']

    print(f"Features: {', '.join(features)}")

    # Train classifier
    rf_results = models.train_rf_classifier(X, y, cv=params['cv_folds'])

    df_top['PREDICTED_TIER'] = rf_results['predictions']
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    print(f"\nOverall accuracy: {rf_results['accuracy']*100:.1f}%")

    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in sorted(df_top['TIER'].unique()):
        sub = df_top[df_top['TIER'] == tier]
        tier_acc = sub['MATCH'].mean()
        print(f"  {tier}: {tier_acc*100:.0f}% ({len(sub)} records)")

    # Feature importance
    importance = models.get_feature_importance(rf_results['model'], features)
    output.save_csv(importance, output_dir, PIPELINE_NAME, 'feature-importance')
    plot_feature_importance(importance, output_dir)

    print("\nFeature importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # Confusion matrix
    tier_order = sorted(df_top['TIER'].unique())
    plot_confusion_matrix(df_top['TIER'], df_top['PREDICTED_TIER'], tier_order, output_dir)

    # -------------------------------------------------------------------------
    # STEP 4: Purchaser analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: PURCHASER TIER VALIDATION")
    print("=" * 70)

    purchaser_analysis = analyze_purchasers(df_top)
    output.save_csv(purchaser_analysis, output_dir, PIPELINE_NAME, 'purchaser-analysis', index=True)

    # Identify misassignments
    misassignments = purchaser_analysis[
        purchaser_analysis['assigned_tier'] != purchaser_analysis['predicted_tier']
    ].copy()

    output.save_csv(misassignments, output_dir, PIPELINE_NAME, 'potential-misassignments', index=True)

    print(f"\nPurchasers where model DISAGREES with assignment: {len(misassignments)}")

    if len(misassignments) > 0:
        print("\nTop potential misassignments:")
        print(misassignments.head(10).to_string())

    # Zero win rate accounts
    zero_wins = purchaser_analysis[purchaser_analysis['win_rate'] == 0]
    print(f"\nPurchasers with 0% win rate: {len(zero_wins)}")
    if len(zero_wins) > 0 and len(zero_wins) <= 10:
        print(zero_wins[['assigned_tier', 'n_quotes', 'avg_estimate', 'avg_margin']].to_string())

    # -------------------------------------------------------------------------
    # STEP 5: Generate recommendations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: RECOMMENDATIONS")
    print("=" * 70)

    recommendations = generate_recommendations(purchaser_analysis, misassignments)
    output.save_csv(recommendations, output_dir, PIPELINE_NAME, 'recommendations')

    # Print high priority recommendations
    high_priority = recommendations[recommendations['priority'] == 'HIGH'] if len(recommendations) > 0 else pd.DataFrame()
    if len(high_priority) > 0:
        print(f"\nHigh priority issues ({len(high_priority)}):")
        for _, row in high_priority.head(5).iterrows():
            print(f"\n  {row['purchaser']}:")
            print(f"    Issue: {row['issue']}")
            print(f"    Currently: {row['current_tier']}")
            print(f"    Model suggests: {row['suggested_tier']}")
            print(f"    Win rate: {row['win_rate']*100:.0f}%, Quotes: {int(row['n_quotes'])}")

    # -------------------------------------------------------------------------
    # STEP 6: Generate report
    # -------------------------------------------------------------------------
    report = generate_report(
        tier_profiles.head(params['n_top_tiers']),
        purchaser_analysis,
        misassignments,
        recommendations,
        rf_results,
        importance,
        params
    )

    with open(f"{output_dir}/{PIPELINE_NAME}-report.txt", 'w') as f:
        f.write(report)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'df_top': df_top,
        'tier_profiles': tier_profiles,
        'purchaser_analysis': purchaser_analysis,
        'misassignments': misassignments,
        'recommendations': recommendations,
        'importance': importance,
        'rf_results': rf_results,
        'output_dir': output_dir,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results = run_pipeline()
