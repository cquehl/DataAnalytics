#!/usr/bin/env python3
"""
Random Forest Classifier Test Script
=====================================

Trains a Random Forest classifier to predict price tiers from order
characteristics. Uses cross-validation to identify potential misassignments.

Parameters:
    data_file     - Path to input CSV
    suppliers     - List of suppliers to filter
    start_date    - Filter from date
    features      - List of features for prediction
    target_col    - Column to predict (default: TIER)
    n_top_tiers   - Number of top tiers to include
    cv_folds      - Number of cross-validation folds

Outputs:
    - Confusion matrix (PNG)
    - Feature importance (CSV, PNG)
    - Per-tier accuracy (CSV)
    - Purchaser analysis (CSV)
    - Potential misassignments (CSV)
    - Text report (TXT)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from efa_core import data, models, viz, output
import config

# =============================================================================
# PARAMETERS
# =============================================================================
DEFAULTS = {
    'data_file': config.DEFAULT_DATA_FILE,
    'suppliers': config.DEFAULT_SUPPLIERS,
    'start_date': config.DEFAULT_START_DATE,
    'features': [
        'ESTIMATETOTALPRICE',
        'ESTIMATETOTALQUANTITY',
        'QTYPERLINEITEM',
        'LINEITEMUNITPRICE',
        'MATERIALUNITWEIGHT',
        'MATERIALCENTWEIGHTCOST',
    ],
    'target_col': 'TIER',
    'n_top_tiers': config.DEFAULT_N_TOP_TIERS,
    'cv_folds': config.RF_CV_FOLDS,
    'output_base': config.DEFAULT_OUTPUT_BASE,
}

TEST_NAME = 'rf-classifier'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_confusion_matrix(conf_matrix, labels, output_dir, test_name):
    """Create confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'confusion-matrix')


def plot_feature_importance(importance_df, output_dir, test_name):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = viz.get_colors()
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color=colors['primary'])

    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Random Forest)')

    # Add value labels
    for bar, imp in zip(bars, importance_df['importance']):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'feature-importance')


def plot_accuracy_by_tier(accuracy_df, output_dir, test_name):
    """Create per-tier accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [viz.get_colors()['significant'] if acc > 0.5 else viz.get_colors()['accent']
              for acc in accuracy_df['accuracy']]

    bars = ax.bar(accuracy_df.index, accuracy_df['accuracy'], color=colors)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% baseline')

    ax.set_xlabel('Tier')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy by Tier\n(green > 50%, red < 50%)')
    ax.legend()

    # Add count labels
    for bar, (_, row) in zip(bars, accuracy_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={int(row["total"])}', ha='center', fontsize=8)

    plt.tight_layout()
    output.save_figure(fig, output_dir, test_name, 'accuracy-by-tier')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(params: dict) -> dict:
    """Run RF classifier analysis pipeline."""
    print("=" * 70)
    print("RANDOM FOREST CLASSIFIER ANALYSIS")
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

    # Filter to top tiers
    top_tiers = df['TIER'].value_counts().head(params['n_top_tiers']).index.tolist()
    df_top = df[df['TIER'].isin(top_tiers)].copy()

    coverage = len(df_top) / len(df) * 100
    print(f"\nTop {params['n_top_tiers']} tiers: {len(df_top):,} records ({coverage:.1f}% of data)")

    # Prepare features
    features = params['features']
    X = df_top[features]
    y = df_top[params['target_col']]

    print(f"Features: {', '.join(features)}")

    # Train classifier
    rf_results = models.train_rf_classifier(X, y, cv=params['cv_folds'])

    # Add predictions to dataframe
    df_top['PREDICTED_TIER'] = rf_results['predictions']
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    # Get feature importance
    importance = models.get_feature_importance(rf_results['model'], features)
    output.save_csv(importance, output_dir, TEST_NAME, 'feature-importance')
    plot_feature_importance(importance, output_dir, TEST_NAME)

    # Evaluate classifier
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    eval_results = models.evaluate_classifier(
        y.values,
        rf_results['predictions'],
        labels=sorted(df_top['TIER'].unique())
    )

    # Save confusion matrix
    conf_df = eval_results['confusion_matrix_df']
    output.save_csv(conf_df, output_dir, TEST_NAME, 'confusion-matrix', index=True)
    plot_confusion_matrix(
        eval_results['confusion_matrix'],
        sorted(df_top['TIER'].unique()),
        output_dir, TEST_NAME
    )

    # Analyze predictions
    pred_analysis = models.analyze_predictions(df_top, 'TIER', 'PREDICTED_TIER')
    accuracy_df = pred_analysis['per_group']
    output.save_csv(accuracy_df, output_dir, TEST_NAME, 'accuracy-by-tier', index=True)
    plot_accuracy_by_tier(accuracy_df, output_dir, TEST_NAME)

    # Analyze purchasers
    print("\n" + "=" * 70)
    print("PURCHASER ANALYSIS")
    print("=" * 70)

    purchaser_analysis = analyze_purchasers(df_top)
    output.save_csv(purchaser_analysis, output_dir, TEST_NAME, 'purchaser-analysis', index=True)

    # Identify misassignments
    disagree = purchaser_analysis[
        purchaser_analysis['assigned_tier'] != purchaser_analysis['predicted_tier']
    ]
    output.save_csv(disagree, output_dir, TEST_NAME, 'potential-misassignments', index=True)

    print(f"\nPurchasers with potential misassignment: {len(disagree)}")
    if len(disagree) > 0:
        print(disagree.to_string())

    # Generate report
    report = generate_report(rf_results, importance, accuracy_df, disagree, params)
    output.save_report(report, output_dir, TEST_NAME)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output.print_summary(output_dir)

    return {
        'df_top': df_top,
        'rf_results': rf_results,
        'importance': importance,
        'purchaser_analysis': purchaser_analysis,
        'output_dir': output_dir,
    }


def analyze_purchasers(df):
    """Aggregate predictions at purchaser level."""
    purchaser_analysis = df.groupby('PURCHASERCOMPANYNAME').agg({
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
    }).round(3)

    purchaser_analysis.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate'
    ]
    return purchaser_analysis.sort_values('n_quotes', ascending=False)


def generate_report(rf_results, importance, accuracy_df, disagree, params):
    """Generate text report."""
    lines = [
        "=" * 70,
        "RANDOM FOREST CLASSIFIER REPORT",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 50,
        f"Data file: {params['data_file']}",
        f"Features: {len(params['features'])}",
        f"Target: {params['target_col']}",
        f"Top tiers: {params['n_top_tiers']}",
        f"CV folds: {params['cv_folds']}",
        "",
        "MODEL PERFORMANCE",
        "-" * 50,
        f"Overall accuracy: {rf_results['accuracy']*100:.1f}%",
        f"CV scores: {rf_results['cv_scores'].round(3)}",
        f"CV mean: {rf_results['cv_scores'].mean():.3f}",
        "",
        "FEATURE IMPORTANCE",
        "-" * 50,
    ]

    for _, row in importance.iterrows():
        lines.append(f"  {row['feature']}: {row['importance']:.3f}")

    lines.extend([
        "",
        "PER-TIER ACCURACY",
        "-" * 50,
        accuracy_df.to_string(),
        "",
        "POTENTIAL MISASSIGNMENTS",
        "-" * 50,
        f"Purchasers with tier mismatch: {len(disagree)}",
    ])

    if len(disagree) > 0:
        lines.append("")
        lines.append(disagree.to_string())

    lines.extend(["", "=" * 70])

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    params = {**DEFAULTS}
    results = run_analysis(params)
