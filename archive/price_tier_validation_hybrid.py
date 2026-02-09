#!/usr/bin/env python3
"""
Price Tier Validation - HYBRID Approach (EFA + Random Forest)
=============================================================

This script combines:
1. EFA (Exploratory Factor Analysis) - for INTERPRETABLE latent factors
2. Random Forest - for ACCURATE prediction of tier assignments

The hybrid approach gives you:
- Factors you can EXPLAIN to stakeholders ("Unit Characteristics", "Order Volume")
- Machine learning ACCURACY to identify potential misassignments
- Best of both worlds: interpretability + predictive power

Prerequisites:
- Run efa_price_tier.py first to generate factor scores

Outputs:
- Console summary of findings
- CSV files in outputs/hybrid_validation/
- Updated findings document
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'Data/Better_Data_Set_Feb3.csv'
FACTOR_SCORES_FILE = 'outputs/efa_tier_analysis/factor_scores.csv'
SUPPLIERS = ['Sabel Steel', 'coosasteel.com']
START_DATE = '2025-08-25'
OUTPUT_DIR = 'outputs/hybrid_validation'
N_TOP_TIERS = 5

# Factor names for interpretability
FACTOR_NAMES = {
    'Factor_1': 'Unit_Characteristics',  # Heavy/expensive units
    'Factor_2': 'Order_Value',           # Big dollar amounts
    'Factor_3': 'Order_Volume'           # High quantities
}


def load_data():
    """Load and merge quote data with factor scores."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    # Load main data
    df = pd.read_csv(DATA_FILE)
    print(f"Raw data: {len(df):,} records")

    # Filter suppliers and dates
    df = df[df['SUPPLIERCOMPANYNAME'].isin(SUPPLIERS)].copy()
    df['DATE'] = pd.to_datetime(df['DATEESTIMATECREATED'])
    df = df[df['DATE'] >= pd.Timestamp(START_DATE)].copy()
    print(f"After filters: {len(df):,} records")

    # Create tier labels
    tier_counts = df['PRICETIERKEY'].value_counts()
    tier_map = {t: f"Tier_{i+1}" for i, t in enumerate(tier_counts.index)}
    df['TIER'] = df['PRICETIERKEY'].map(tier_map)

    # Load factor scores (REQUIRED - this is a hybrid approach)
    if not os.path.exists(FACTOR_SCORES_FILE):
        raise FileNotFoundError(
            f"\nFactor scores not found at: {FACTOR_SCORES_FILE}\n\n"
            f"This hybrid analysis REQUIRES factor scores from EFA.\n"
            f"Please run: python efa_price_tier.py\n"
        )

    factor_scores = pd.read_csv(FACTOR_SCORES_FILE, index_col=0)
    print(f"Loaded factor scores: {len(factor_scores):,} records")

    # Rename factors for interpretability
    factor_scores = factor_scores.rename(columns=FACTOR_NAMES)

    # Merge factor scores with main data
    df = df.join(factor_scores, how='inner')
    print(f"After merge: {len(df):,} records with factor scores")

    return df, factor_scores


def prepare_features(df, use_factors=True, use_raw=False):
    """
    Prepare feature matrix for model training.

    Options:
    - use_factors=True: Use EFA factor scores (interpretable)
    - use_raw=True: Use raw numeric variables
    - Both: Hybrid with all features
    """
    print("\n" + "=" * 70)
    print("STEP 2: PREPARING FEATURES")
    print("=" * 70)

    features = []
    feature_names = []

    if use_factors:
        factor_cols = list(FACTOR_NAMES.values())
        available_factors = [c for c in factor_cols if c in df.columns]
        if available_factors:
            features.append(df[available_factors].values)
            feature_names.extend(available_factors)
            print(f"Factor features: {available_factors}")

    if use_raw:
        raw_cols = [
            'ESTIMATETOTALPRICE', 'ESTIMATETOTALQUANTITY', 'QTYPERLINEITEM',
            'LINEITEMUNITPRICE', 'MATERIALUNITWEIGHT', 'MATERIALCENTWEIGHTCOST'
        ]
        available_raw = [c for c in raw_cols if c in df.columns]
        if available_raw:
            # Standardize raw features
            scaler = StandardScaler()
            raw_scaled = scaler.fit_transform(df[available_raw].fillna(0))
            features.append(raw_scaled)
            feature_names.extend([f"raw_{c}" for c in available_raw])
            print(f"Raw features: {available_raw}")

    X = np.hstack(features) if len(features) > 1 else features[0]
    print(f"\nTotal features: {len(feature_names)}")

    return X, feature_names


def train_and_validate(df, X, feature_names):
    """Train Random Forest using factor scores and validate tier assignments."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING & VALIDATION")
    print("=" * 70)

    # Focus on top tiers
    tier_counts = df['TIER'].value_counts()
    top_tiers = tier_counts.head(N_TOP_TIERS).index.tolist()
    mask = df['TIER'].isin(top_tiers)

    X_top = X[mask]
    df_top = df[mask].copy()
    y = df_top['TIER']

    coverage = len(df_top) / len(df) * 100
    print(f"Using top {N_TOP_TIERS} tiers: {len(df_top):,} records ({coverage:.0f}% of data)")

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_leaf=10
    )

    # Cross-validation
    cv_scores = cross_val_score(rf, X_top, y_encoded, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    # Get predictions for all records
    y_pred = cross_val_predict(rf, X_top, y_encoded, cv=5)
    y_pred_labels = le.inverse_transform(y_pred)

    df_top['PREDICTED_TIER'] = y_pred_labels
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    # Overall accuracy
    accuracy = df_top['MATCH'].mean()
    print(f"Overall accuracy: {accuracy*100:.1f}%")

    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in sorted(df_top['TIER'].unique()):
        sub = df_top[df_top['TIER'] == tier]
        tier_acc = sub['MATCH'].mean()
        print(f"  {tier}: {tier_acc*100:.0f}% ({len(sub)} records)")

    # Feature importance
    rf.fit(X_top, y_encoded)
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s}: {row['importance']:.3f} {bar}")

    return df_top, importance, rf, le


def analyze_purchasers(df_top):
    """Analyze tier assignments at the purchaser level."""
    print("\n" + "=" * 70)
    print("STEP 4: PURCHASER ANALYSIS")
    print("=" * 70)

    # Get factor columns
    factor_cols = [c for c in FACTOR_NAMES.values() if c in df_top.columns]

    # Aggregate by purchaser
    agg_dict = {
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
        'MARGINSYSTEMRECOMMENDED': 'mean'
    }

    # Add factor columns to aggregation
    for fc in factor_cols:
        agg_dict[fc] = 'mean'

    purchaser_analysis = df_top.groupby('PURCHASERCOMPANYNAME').agg(agg_dict).round(3)

    # Flatten column names
    purchaser_analysis.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate', 'avg_margin'
    ] + [f'avg_{fc}' for fc in factor_cols]

    purchaser_analysis = purchaser_analysis.sort_values('n_quotes', ascending=False)

    # Identify disagreements
    disagree = purchaser_analysis[
        purchaser_analysis['assigned_tier'] != purchaser_analysis['predicted_tier']
    ].copy()

    print(f"\nPurchasers where model DISAGREES: {len(disagree)}")
    if len(disagree) > 0:
        print("\n" + disagree.to_string())

    # Zero win rate accounts
    zero_wins = purchaser_analysis[purchaser_analysis['win_rate'] == 0]
    print(f"\n\nPurchasers with 0% win rate: {len(zero_wins)}")

    return purchaser_analysis, disagree


def generate_recommendations(purchaser_analysis, disagree, factor_cols):
    """Generate actionable recommendations with factor-based explanations."""
    print("\n" + "=" * 70)
    print("STEP 5: RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    for idx, row in disagree.iterrows():
        rec = {
            'purchaser': idx,
            'assigned_tier': row['assigned_tier'],
            'predicted_tier': row['predicted_tier'],
            'win_rate': row['win_rate'],
            'n_quotes': row['n_quotes'],
            'avg_estimate': row['avg_estimate']
        }

        # Add factor scores for explanation
        for fc in factor_cols:
            col_name = f'avg_{fc}'
            if col_name in row.index:
                rec[fc] = row[col_name]

        # Determine issue type
        if row['win_rate'] == 0:
            rec['issue'] = 'NEVER WINS - review pricing'
        elif row['win_rate'] < 0.15:
            rec['issue'] = 'LOW WIN RATE - potential overpricing'
        elif row['win_rate'] > 0.85:
            rec['issue'] = 'ALWAYS WINS - leaving money on table?'
        else:
            rec['issue'] = 'Factor profile mismatch'

        recommendations.append(rec)

        print(f"\n{idx}:")
        print(f"  Issue: {rec['issue']}")
        print(f"  Currently: {row['assigned_tier']} → Model suggests: {row['predicted_tier']}")
        print(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {int(row['n_quotes'])}")

        # Explain using factors
        print(f"  Factor profile:")
        for fc in factor_cols:
            col_name = f'avg_{fc}'
            if col_name in row.index:
                val = row[col_name]
                level = "HIGH" if val > 0.3 else "LOW" if val < -0.3 else "avg"
                print(f"    - {fc}: {val:+.2f} ({level})")

    return pd.DataFrame(recommendations)


def plot_results(df_top, importance, output_dir):
    """Generate visualizations."""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZATIONS")
    print("=" * 70)

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    colors = ['#3498db' if 'raw' not in f else '#95a5a6' for f in importance['feature']]
    plt.barh(importance['feature'], importance['importance'], color=colors)
    plt.xlabel('Importance')
    plt.title('Feature Importance (blue = EFA factors, gray = raw features)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()
    print(f"Feature importance plot saved to: {output_dir}/feature_importance.png")

    # Confusion matrix
    tier_order = df_top['TIER'].value_counts().index.tolist()
    cm = confusion_matrix(df_top['TIER'], df_top['PREDICTED_TIER'], labels=tier_order)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tier_order, yticklabels=tier_order)
    plt.xlabel('Predicted Tier')
    plt.ylabel('Actual Tier')
    plt.title('Confusion Matrix: Actual vs Predicted Tier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")

    # Factor scores by tier (if available)
    factor_cols = [c for c in FACTOR_NAMES.values() if c in df_top.columns]
    if factor_cols:
        fig, axes = plt.subplots(1, len(factor_cols), figsize=(5*len(factor_cols), 6))
        if len(factor_cols) == 1:
            axes = [axes]

        for ax, factor in zip(axes, factor_cols):
            tier_order = df_top['TIER'].value_counts().index.tolist()
            sns.boxplot(data=df_top, x='TIER', y=factor, order=tier_order, ax=ax)
            ax.set_title(f'{factor} by Tier')
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/factors_by_tier.png", dpi=150)
        plt.close()
        print(f"Factor distributions saved to: {output_dir}/factors_by_tier.png")


def save_outputs(purchaser_analysis, disagree, recommendations, importance, df_top, output_dir):
    """Save all results to CSV files."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVING OUTPUTS")
    print("=" * 70)

    purchaser_analysis.to_csv(f"{output_dir}/purchaser_analysis.csv")
    print(f"  {output_dir}/purchaser_analysis.csv")

    disagree.to_csv(f"{output_dir}/potential_misassignments.csv")
    print(f"  {output_dir}/potential_misassignments.csv")

    if len(recommendations) > 0:
        recommendations.to_csv(f"{output_dir}/recommendations.csv", index=False)
        print(f"  {output_dir}/recommendations.csv")

    importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    print(f"  {output_dir}/feature_importance.csv")

    # Save detailed quote-level results
    output_cols = ['PURCHASERCOMPANYNAME', 'TIER', 'PREDICTED_TIER', 'MATCH',
                   'ISWON', 'ESTIMATETOTALPRICE', 'MARGINSYSTEMRECOMMENDED']
    factor_cols = [c for c in FACTOR_NAMES.values() if c in df_top.columns]
    output_cols.extend(factor_cols)

    df_top[output_cols].to_csv(f"{output_dir}/quote_level_results.csv", index=False)
    print(f"  {output_dir}/quote_level_results.csv")


def generate_summary_report(purchaser_analysis, disagree, importance, accuracy, output_dir):
    """Generate a human-readable summary report."""

    factor_cols = [c for c in FACTOR_NAMES.values()]

    report = []
    report.append("=" * 70)
    report.append("HYBRID PRICE TIER VALIDATION - SUMMARY REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("METHODOLOGY")
    report.append("-" * 50)
    report.append("""
This analysis combines two approaches:

1. EXPLORATORY FACTOR ANALYSIS (EFA)
   - Discovered 3 latent factors in order characteristics
   - Factors are INTERPRETABLE and can be explained to stakeholders

2. RANDOM FOREST CLASSIFIER
   - Uses factor scores to predict tier assignments
   - Identifies quotes/purchasers that don't match their tier's profile

The hybrid approach gives you:
   - INTERPRETABILITY from EFA (what do the factors mean?)
   - ACCURACY from Random Forest (who might be misassigned?)
""")

    report.append("")
    report.append("THE THREE FACTORS")
    report.append("-" * 50)
    report.append("""
Factor 1: UNIT CHARACTERISTICS
   - High scores = heavy units, expensive per-unit price
   - Low scores = light units, cheap per-unit price

Factor 2: ORDER VALUE
   - High scores = large dollar amount orders
   - Low scores = small dollar amount orders

Factor 3: ORDER VOLUME
   - High scores = high quantity orders
   - Low scores = low quantity orders
""")

    report.append("")
    report.append("MODEL PERFORMANCE")
    report.append("-" * 50)
    report.append(f"Overall accuracy: {accuracy*100:.1f}%")
    report.append("")
    report.append("Feature importance:")
    for _, row in importance.iterrows():
        report.append(f"  {row['feature']:25s}: {row['importance']:.3f}")

    report.append("")
    report.append("")
    report.append("POTENTIAL MISASSIGNMENTS")
    report.append("-" * 50)
    report.append(f"Purchasers flagged: {len(disagree)}")
    report.append("")

    for idx, row in disagree.iterrows():
        report.append(f"{idx}:")
        report.append(f"  Assigned: {row['assigned_tier']} → Suggested: {row['predicted_tier']}")
        report.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {int(row['n_quotes'])}")
        report.append("")

    report.append("")
    report.append("=" * 70)
    report.append("HOW TO USE THESE FINDINGS")
    report.append("=" * 70)
    report.append("""
1. REVIEW FLAGGED PURCHASERS
   Look at purchasers where the model disagrees with assignment.
   Especially focus on those with 0% or very low win rates.

2. UNDERSTAND FACTOR PROFILES
   Each tier has a typical "profile" based on factor scores.
   Mismatches suggest the purchaser's orders don't fit their tier.

3. EXPLAIN TO STAKEHOLDERS
   "This customer's orders are mostly small quantities of heavy,
   expensive materials (high Unit_Characteristics, low Order_Volume).
   That profile typically maps to Tier_X, but they're in Tier_Y."

4. ITERATE
   Add more data, refine tiers, and re-run the analysis.
   The hybrid approach will continue to identify outliers.
""")

    report_text = "\n".join(report)

    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write(report_text)

    print(f"\nSummary report saved to: {output_dir}/summary_report.txt")

    return report_text


def main():
    """Run the hybrid price tier validation."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data (factor_scores guaranteed by FileNotFoundError)
    df, factor_scores = load_data()

    # Step 2: Prepare features using EFA factors
    X, feature_names = prepare_features(df, use_factors=True, use_raw=False)

    # Step 3: Train and validate
    df_top, importance, rf, le = train_and_validate(df, X, feature_names)
    accuracy = df_top['MATCH'].mean()

    # Step 4: Analyze purchasers
    factor_cols = [c for c in FACTOR_NAMES.values() if c in df_top.columns]
    purchaser_analysis, disagree = analyze_purchasers(df_top)

    # Step 5: Generate recommendations
    recommendations = generate_recommendations(purchaser_analysis, disagree, factor_cols)

    # Step 6: Visualizations
    plot_results(df_top, importance, OUTPUT_DIR)

    # Step 7: Save outputs
    save_outputs(purchaser_analysis, disagree, recommendations, importance, df_top, OUTPUT_DIR)

    # Generate summary report
    generate_summary_report(purchaser_analysis, disagree, importance, accuracy, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
