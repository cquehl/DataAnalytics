#!/usr/bin/env python3
"""
Price Tier Validation Analysis
==============================
Validates whether price tier assignments make sense based on order characteristics.

Methodology:
- Uses a Random Forest classifier to predict tiers from order characteristics
- Compares predicted tiers vs actual assignments to find potential misclassifications
- Identifies purchasers whose order patterns don't match their assigned tier

Output:
- Console summary of findings
- CSV files in outputs/ directory for further analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# Create outputs directory
os.makedirs('outputs', exist_ok=True)


def load_and_filter_data(filepath, suppliers, start_date):
    """Load data and filter to specified suppliers and date range."""
    df = pd.read_csv(filepath)

    # Filter suppliers
    df = df[df['SUPPLIERCOMPANYNAME'].isin(suppliers)].copy()

    # Parse and filter dates
    df['DATE'] = pd.to_datetime(df['DATEESTIMATECREATED'])
    df = df[df['DATE'] >= pd.Timestamp(start_date)].copy()

    return df


def create_tier_labels(df):
    """Create readable tier labels based on frequency."""
    tier_counts = df['PRICETIERKEY'].value_counts()
    tier_map = {t: f"Tier_{i+1}" for i, t in enumerate(tier_counts.index)}
    df['TIER'] = df['PRICETIERKEY'].map(tier_map)
    return df, tier_map


def profile_tiers(df):
    """Generate tier profiles showing key metrics."""
    print("\n" + "=" * 80)
    print("TIER PROFILES")
    print("=" * 80)

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
            'avg_margin': sub['MARGINSYSTEMRECOMMENDED'].mean()
        }
        profiles.append(profile)

        print(f"\n{tier} ({profile['n_quotes']} quotes, {profile['win_rate']*100:.0f}% win rate):")
        print(f"  Purchasers: {profile['n_purchasers']}")
        print(f"  Avg estimate: ${profile['avg_estimate']:,.0f}")
        print(f"  Avg quantity: {profile['avg_quantity']:.1f}")
        print(f"  Avg unit price: ${profile['avg_unit_price']:.2f}")
        print(f"  Avg margin: {profile['avg_margin']:.1f}%")

    return pd.DataFrame(profiles)


def train_and_validate(df, n_top_tiers=5):
    """
    Train Random Forest classifier and validate tier assignments.

    Uses cross-validation to get predictions on held-out data,
    avoiding overfitting.
    """
    # Focus on top tiers (covers most of data)
    tier_counts = df['PRICETIERKEY'].value_counts()
    top_tiers = tier_counts.head(n_top_tiers).index.tolist()
    df_top = df[df['PRICETIERKEY'].isin(top_tiers)].copy()

    coverage = len(df_top) / len(df) * 100
    print(f"\n" + "=" * 80)
    print(f"MODEL TRAINING (Top {n_top_tiers} tiers, {len(df_top)} records, {coverage:.0f}% of data)")
    print("=" * 80)

    # Features: order characteristics only (no purchaser identity)
    numeric_features = [
        'ESTIMATETOTALPRICE',
        'ESTIMATETOTALQUANTITY',
        'QTYPERLINEITEM',
        'LINEITEMUNITPRICE',
        'MATERIALUNITWEIGHT',
        'MATERIALCENTWEIGHTCOST'
    ]

    print(f"\nFeatures used: {', '.join(numeric_features)}")

    X = df_top[numeric_features].fillna(0)
    y = df_top['TIER']

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest with cross-validation predictions
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_leaf=10
    )

    # Cross-validation predictions (5-fold)
    y_pred = cross_val_predict(rf, X_scaled, y_encoded, cv=5)
    y_pred_labels = le.inverse_transform(y_pred)

    df_top['PREDICTED_TIER'] = y_pred_labels
    df_top['MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    # Overall accuracy
    accuracy = df_top['MATCH'].mean()
    print(f"\nOverall accuracy: {accuracy*100:.1f}%")

    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in sorted(df_top['TIER'].unique()):
        sub = df_top[df_top['TIER'] == tier]
        tier_acc = sub['MATCH'].mean()
        n_records = len(sub)
        print(f"  {tier}: {tier_acc*100:.0f}% ({n_records} records)")

    # Feature importance
    rf.fit(X_scaled, y_encoded)  # Refit on all data for feature importance
    importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return df_top, importance


def analyze_purchasers(df_top):
    """Analyze tier assignments at the purchaser level."""
    print("\n" + "=" * 80)
    print("PURCHASER TIER VALIDATION")
    print("=" * 80)

    purchaser_analysis = df_top.groupby('PURCHASERCOMPANYNAME').agg({
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
        'MARGINSYSTEMRECOMMENDED': 'mean'
    }).round(3)

    purchaser_analysis.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate', 'avg_margin'
    ]
    purchaser_analysis = purchaser_analysis.sort_values('n_quotes', ascending=False)

    # Identify disagreements
    disagree = purchaser_analysis[
        purchaser_analysis['assigned_tier'] != purchaser_analysis['predicted_tier']
    ].copy()

    print(f"\nPurchasers where model DISAGREES with assignment: {len(disagree)}")
    if len(disagree) > 0:
        print("\n" + disagree.to_string())

    # Identify 0% win rate accounts
    zero_wins = purchaser_analysis[purchaser_analysis['win_rate'] == 0]
    print(f"\n\nPurchasers with 0% win rate: {len(zero_wins)}")
    if len(zero_wins) > 0:
        print(zero_wins[['assigned_tier', 'n_quotes', 'avg_estimate', 'avg_margin']].to_string())

    # Identify 100% win rate accounts
    all_wins = purchaser_analysis[purchaser_analysis['win_rate'] == 1.0]
    print(f"\n\nPurchasers with 100% win rate: {len(all_wins)}")
    if len(all_wins) > 0:
        print(all_wins[['assigned_tier', 'n_quotes', 'avg_estimate', 'avg_margin']].to_string())

    return purchaser_analysis, disagree


def generate_recommendations(purchaser_analysis, disagree):
    """Generate actionable recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Low win rate in wrong tier
    low_win_disagree = disagree[disagree['win_rate'] < 0.05]
    for idx, row in low_win_disagree.iterrows():
        rec = {
            'purchaser': idx,
            'issue': 'Low win rate, possible misassignment',
            'current_tier': row['assigned_tier'],
            'suggested_tier': row['predicted_tier'],
            'win_rate': row['win_rate'],
            'n_quotes': row['n_quotes']
        }
        recommendations.append(rec)
        print(f"\n{idx}:")
        print(f"  Issue: Low win rate ({row['win_rate']*100:.0f}%), model suggests different tier")
        print(f"  Currently: {row['assigned_tier']}")
        print(f"  Model suggests: {row['predicted_tier']}")
        print(f"  Quotes: {int(row['n_quotes'])}, Avg estimate: ${row['avg_estimate']:,.0f}")

    # High win rate - leaving money on table?
    high_win = purchaser_analysis[purchaser_analysis['win_rate'] > 0.3]
    for idx, row in high_win.iterrows():
        if row['n_quotes'] >= 5:  # Only flag if enough data
            rec = {
                'purchaser': idx,
                'issue': 'Very high win rate - review pricing',
                'current_tier': row['assigned_tier'],
                'suggested_tier': 'Consider higher tier',
                'win_rate': row['win_rate'],
                'n_quotes': row['n_quotes']
            }
            recommendations.append(rec)

    return pd.DataFrame(recommendations)


def save_outputs(tier_profiles, purchaser_analysis, disagree, recommendations, importance):
    """Save all analysis outputs to CSV files."""
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    tier_profiles.to_csv('outputs/tier_profiles.csv', index=False)
    print("  outputs/tier_profiles.csv")

    purchaser_analysis.to_csv('outputs/purchaser_tier_analysis.csv')
    print("  outputs/purchaser_tier_analysis.csv")

    disagree.to_csv('outputs/potential_misassignments.csv')
    print("  outputs/potential_misassignments.csv")

    if len(recommendations) > 0:
        recommendations.to_csv('outputs/tier_recommendations.csv', index=False)
        print("  outputs/tier_recommendations.csv")

    importance.to_csv('outputs/feature_importance.csv', index=False)
    print("  outputs/feature_importance.csv")


def main():
    """Run the full price tier validation analysis."""
    print("=" * 80)
    print("PRICE TIER VALIDATION ANALYSIS")
    print("=" * 80)

    # Configuration
    DATA_FILE = 'Data/Better_Data_Set_Feb3.csv'
    SUPPLIERS = ['Sabel Steel', 'coosasteel.com']
    START_DATE = '2025-08-25'
    N_TOP_TIERS = 5  # Focus on top N tiers

    print(f"\nData file: {DATA_FILE}")
    print(f"Suppliers: {', '.join(SUPPLIERS)}")
    print(f"Date range: {START_DATE} onwards")

    # Load and filter data
    df = load_and_filter_data(DATA_FILE, SUPPLIERS, START_DATE)
    df, tier_map = create_tier_labels(df)

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique tiers: {df['PRICETIERKEY'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")
    print(f"Overall win rate: {df['ISWON'].mean()*100:.1f}%")

    # Profile tiers
    tier_profiles = profile_tiers(df)

    # Train model and validate
    df_top, importance = train_and_validate(df, n_top_tiers=N_TOP_TIERS)

    # Analyze purchasers
    purchaser_analysis, disagree = analyze_purchasers(df_top)

    # Generate recommendations
    recommendations = generate_recommendations(purchaser_analysis, disagree)

    # Save outputs
    save_outputs(tier_profiles, purchaser_analysis, disagree, recommendations, importance)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSee 'Initial Price Tier Findings.txt' for detailed methodology explanation.")


if __name__ == '__main__':
    main()
