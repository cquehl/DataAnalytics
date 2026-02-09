#!/usr/bin/env python3
"""
Sabel Steel Price Tier Outlier Analysis
========================================

Focused analysis on Sabel Steel to identify purchasers who may be
assigned to the wrong price tier based on their order characteristics.

Methodology:
1. EFA - Extract interpretable latent factors from order data
2. Random Forest - Predict expected tier based on order patterns
3. Outlier Detection - Flag purchasers where assigned ≠ predicted tier

Output: Actionable list of purchasers to review for potential tier reassignment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
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
DATA_FILE = 'Data/Decent_Data_Set_Feb3.csv'
SUPPLIER = 'Sabel Steel'  # Focus on Sabel only
OUTPUT_DIR = 'outputs/sabel_outliers'
N_TOP_TIERS = 5  # Focus on tiers with enough data

# Features for EFA (order characteristics - NO margins or tier info!)
EFA_FEATURES = [
    'ESTIMATETOTALPRICE',
    'ESTIMATETOTALQUANTITY',
    'QTYPERLINEITEM',
    'LINEITEMUNITPRICE',
    'LINEITEMTOTALPRICE',
    'MATERIALUNITWEIGHT',
    'MATERIALCENTWEIGHTCOST'
]


def load_sabel_data():
    """Load data filtered to Sabel Steel only."""
    print("=" * 70)
    print("STEP 1: LOADING SABEL STEEL DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    print(f"Raw data: {len(df):,} records")

    # Filter to Sabel Steel only
    df = df[df['SUPPLIERCOMPANYNAME'] == SUPPLIER].copy()
    print(f"Sabel Steel records: {len(df):,}")

    # Create tier labels based on frequency
    tier_counts = df['PRICETIERKEY'].value_counts()
    tier_map = {t: f"Tier_{i+1}" for i, t in enumerate(tier_counts.index)}
    df['TIER'] = df['PRICETIERKEY'].map(tier_map)

    print(f"\nPrice tier distribution:")
    for tier, count in df['TIER'].value_counts().head(N_TOP_TIERS + 2).items():
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} records ({pct:.1f}%)")

    print(f"\nUnique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")
    print(f"Overall win rate: {df['ISWON'].mean()*100:.1f}%")

    return df, tier_map


def run_efa(df):
    """Run Exploratory Factor Analysis on order characteristics."""
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY FACTOR ANALYSIS")
    print("=" * 70)

    # Prepare data for EFA
    available_features = [f for f in EFA_FEATURES if f in df.columns]
    efa_data = df[available_features].dropna()
    print(f"Records with complete data: {len(efa_data):,}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(efa_data)
    X_df = pd.DataFrame(X_scaled, columns=available_features, index=efa_data.index)

    # Check factorability
    chi2, p_bartlett = calculate_bartlett_sphericity(X_df)
    kmo_all, kmo_model = calculate_kmo(X_df)

    print(f"\nFactorability tests:")
    print(f"  Bartlett's test: χ² = {chi2:.1f}, p < 0.001 ✓" if p_bartlett < 0.001 else f"  Bartlett's test: p = {p_bartlett:.4f}")
    print(f"  KMO: {kmo_model:.3f} {'✓' if kmo_model > 0.6 else '(borderline)'}")

    # Determine number of factors (Kaiser criterion)
    fa_initial = FactorAnalyzer(n_factors=len(available_features), rotation=None)
    fa_initial.fit(X_df)
    eigenvalues = fa_initial.get_eigenvalues()[0]
    n_factors = sum(eigenvalues > 1)
    print(f"\nEigenvalues > 1: {n_factors} factors")

    # Extract factors with Varimax rotation
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(X_df)

    # Get loadings
    loadings = pd.DataFrame(
        fa.loadings_,
        index=available_features,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    print(f"\nFactor loadings (Varimax rotation):")
    print(loadings.round(3).to_string())

    # Interpret factors
    print("\n" + "-" * 50)
    print("FACTOR INTERPRETATION:")
    print("-" * 50)

    factor_names = {}
    for col in loadings.columns:
        top_loadings = loadings[col].abs().nlargest(2)
        vars_loaded = top_loadings.index.tolist()

        # Heuristic naming based on which variables load
        if 'MATERIALUNITWEIGHT' in vars_loaded or 'LINEITEMUNITPRICE' in vars_loaded:
            name = 'Unit_Characteristics'
        elif 'ESTIMATETOTALPRICE' in vars_loaded or 'LINEITEMTOTALPRICE' in vars_loaded:
            name = 'Order_Value'
        elif 'ESTIMATETOTALQUANTITY' in vars_loaded or 'QTYPERLINEITEM' in vars_loaded:
            name = 'Order_Volume'
        else:
            name = col

        factor_names[col] = name
        print(f"  {col} → {name}")
        for var in vars_loaded:
            print(f"      {var}: {loadings.loc[var, col]:.3f}")

    # Calculate factor scores
    factor_scores = pd.DataFrame(
        fa.transform(X_df),
        index=efa_data.index,
        columns=[factor_names.get(f'Factor_{i+1}', f'Factor_{i+1}') for i in range(n_factors)]
    )

    # Variance explained
    variance = fa.get_factor_variance()
    total_var = sum(variance[1]) * 100
    print(f"\nTotal variance explained: {total_var:.1f}%")

    return factor_scores, loadings, fa, scaler, factor_names


def train_tier_predictor(df, factor_scores):
    """Train Random Forest to predict tier from factor scores."""
    print("\n" + "=" * 70)
    print("STEP 3: TIER PREDICTION MODEL")
    print("=" * 70)

    # Merge factor scores with main data
    df_with_factors = df.join(factor_scores, how='inner')
    print(f"Records with factor scores: {len(df_with_factors):,}")

    # Focus on top tiers (enough data for reliable prediction)
    tier_counts = df_with_factors['TIER'].value_counts()
    top_tiers = tier_counts.head(N_TOP_TIERS).index.tolist()

    df_top = df_with_factors[df_with_factors['TIER'].isin(top_tiers)].copy()
    coverage = len(df_top) / len(df_with_factors) * 100
    print(f"Using top {N_TOP_TIERS} tiers: {len(df_top):,} records ({coverage:.0f}%)")

    # Prepare features and target
    factor_cols = factor_scores.columns.tolist()
    X = df_top[factor_cols].values
    y = df_top['TIER']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y_encoded, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    # Get predictions
    y_pred = cross_val_predict(rf, X, y_encoded, cv=5)
    y_pred_labels = le.inverse_transform(y_pred)

    df_top['PREDICTED_TIER'] = y_pred_labels
    df_top['TIER_MATCH'] = df_top['TIER'] == df_top['PREDICTED_TIER']

    accuracy = df_top['TIER_MATCH'].mean()
    print(f"Overall accuracy: {accuracy*100:.1f}%")

    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in sorted(df_top['TIER'].unique()):
        tier_data = df_top[df_top['TIER'] == tier]
        tier_acc = tier_data['TIER_MATCH'].mean()
        print(f"  {tier}: {tier_acc*100:.0f}% ({len(tier_data)} records)")

    # Feature importance
    rf.fit(X, y_encoded)
    importance = pd.DataFrame({
        'factor': factor_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFactor importance for tier prediction:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"  {row['factor']:25s}: {row['importance']:.3f} {bar}")

    return df_top, rf, le, importance


def identify_outliers(df_top, factor_cols):
    """Identify purchasers who may be in the wrong tier."""
    print("\n" + "=" * 70)
    print("STEP 4: OUTLIER IDENTIFICATION")
    print("=" * 70)

    # Aggregate by purchaser
    agg_dict = {
        'TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'PREDICTED_TIER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'TIER_MATCH': 'mean',
        'ISWON': ['mean', 'count'],
        'ESTIMATETOTALPRICE': 'mean',
        'MARGINSYSTEMRECOMMENDED': 'mean'
    }

    # Add factor columns
    for fc in factor_cols:
        agg_dict[fc] = 'mean'

    purchaser_stats = df_top.groupby('PURCHASERCOMPANYNAME').agg(agg_dict).round(3)

    # Flatten column names
    purchaser_stats.columns = [
        'assigned_tier', 'predicted_tier', 'match_rate',
        'win_rate', 'n_quotes', 'avg_estimate', 'avg_margin'
    ] + [f'avg_{fc}' for fc in factor_cols]

    purchaser_stats = purchaser_stats.sort_values('n_quotes', ascending=False)

    # Identify disagreements (OUTLIERS)
    outliers = purchaser_stats[
        purchaser_stats['assigned_tier'] != purchaser_stats['predicted_tier']
    ].copy()

    print(f"\nTotal purchasers analyzed: {len(purchaser_stats)}")
    print(f"POTENTIAL OUTLIERS (tier mismatch): {len(outliers)}")

    # Categorize outliers
    print("\n" + "-" * 50)
    print("OUTLIER DETAILS:")
    print("-" * 50)

    for idx, row in outliers.iterrows():
        print(f"\n📍 {idx}")
        print(f"   Assigned: {row['assigned_tier']} → Model predicts: {row['predicted_tier']}")
        print(f"   Quotes: {int(row['n_quotes'])}, Win rate: {row['win_rate']*100:.0f}%")
        print(f"   Avg estimate: ${row['avg_estimate']:,.0f}")

        # Flag special cases
        flags = []
        if row['win_rate'] == 0:
            flags.append("🔴 NEVER WINS")
        elif row['win_rate'] < 0.15:
            flags.append("🟠 LOW WIN RATE")
        elif row['win_rate'] > 0.80:
            flags.append("🟢 HIGH WIN RATE (leaving $$ on table?)")

        if flags:
            print(f"   Flags: {', '.join(flags)}")

        # Factor profile
        print(f"   Factor profile:")
        for fc in factor_cols:
            val = row[f'avg_{fc}']
            level = "HIGH" if val > 0.5 else "LOW" if val < -0.5 else "avg"
            bar = "+" * int(max(0, val) * 10) if val > 0 else "-" * int(abs(min(0, val)) * 10)
            print(f"      {fc}: {val:+.2f} ({level}) {bar}")

    return purchaser_stats, outliers


def generate_report(purchaser_stats, outliers, importance, accuracy, factor_cols, output_dir):
    """Generate comprehensive outlier report."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING REPORT")
    print("=" * 70)

    # Create detailed recommendations
    recommendations = []

    for idx, row in outliers.iterrows():
        rec = {
            'purchaser': idx,
            'assigned_tier': row['assigned_tier'],
            'predicted_tier': row['predicted_tier'],
            'n_quotes': int(row['n_quotes']),
            'win_rate': row['win_rate'],
            'avg_estimate': row['avg_estimate'],
            'severity': 'LOW'
        }

        # Determine severity and recommendation
        if row['win_rate'] == 0:
            rec['severity'] = 'HIGH'
            rec['recommendation'] = f"URGENT: Review pricing. 0% win rate suggests significant overpricing. Consider moving to {row['predicted_tier']}."
        elif row['win_rate'] < 0.15:
            rec['severity'] = 'HIGH'
            rec['recommendation'] = f"Review pricing. Low win rate ({row['win_rate']*100:.0f}%) suggests potential overpricing. Consider {row['predicted_tier']}."
        elif row['win_rate'] > 0.80:
            rec['severity'] = 'MEDIUM'
            rec['recommendation'] = f"Revenue opportunity. High win rate ({row['win_rate']*100:.0f}%) may indicate underpricing. Consider moving from {row['assigned_tier']} toward {row['predicted_tier']}."
        else:
            rec['severity'] = 'LOW'
            rec['recommendation'] = f"Factor profile suggests {row['predicted_tier']} may be a better fit than current {row['assigned_tier']}."

        # Add factor explanations
        for fc in factor_cols:
            rec[fc] = row[f'avg_{fc}']

        recommendations.append(rec)

    recommendations_df = pd.DataFrame(recommendations)

    # Sort by severity
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    if len(recommendations_df) > 0:
        recommendations_df['_sort'] = recommendations_df['severity'].map(severity_order)
        recommendations_df = recommendations_df.sort_values(['_sort', 'n_quotes'], ascending=[True, False])
        recommendations_df = recommendations_df.drop('_sort', axis=1)

    # Save files
    purchaser_stats.to_csv(f"{output_dir}/all_purchasers.csv")
    outliers.to_csv(f"{output_dir}/outliers.csv")
    recommendations_df.to_csv(f"{output_dir}/recommendations.csv", index=False)
    importance.to_csv(f"{output_dir}/factor_importance.csv", index=False)

    print(f"Saved: {output_dir}/all_purchasers.csv")
    print(f"Saved: {output_dir}/outliers.csv")
    print(f"Saved: {output_dir}/recommendations.csv")
    print(f"Saved: {output_dir}/factor_importance.csv")

    # Generate text summary
    summary = []
    summary.append("=" * 70)
    summary.append("SABEL STEEL PRICE TIER OUTLIER ANALYSIS")
    summary.append("=" * 70)
    summary.append("")
    summary.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"Model Accuracy: {accuracy*100:.1f}%")
    summary.append(f"Total Purchasers: {len(purchaser_stats)}")
    summary.append(f"Outliers Identified: {len(outliers)}")
    summary.append("")
    summary.append("-" * 70)
    summary.append("HIGH SEVERITY OUTLIERS (Requires Immediate Review)")
    summary.append("-" * 70)

    high_sev = recommendations_df[recommendations_df['severity'] == 'HIGH'] if len(recommendations_df) > 0 else pd.DataFrame()
    if len(high_sev) > 0:
        for _, row in high_sev.iterrows():
            summary.append(f"\n{row['purchaser']}")
            summary.append(f"  {row['assigned_tier']} → {row['predicted_tier']}")
            summary.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
            summary.append(f"  💡 {row['recommendation']}")
    else:
        summary.append("  None identified.")

    summary.append("")
    summary.append("-" * 70)
    summary.append("MEDIUM SEVERITY OUTLIERS (Revenue Opportunities)")
    summary.append("-" * 70)

    med_sev = recommendations_df[recommendations_df['severity'] == 'MEDIUM'] if len(recommendations_df) > 0 else pd.DataFrame()
    if len(med_sev) > 0:
        for _, row in med_sev.iterrows():
            summary.append(f"\n{row['purchaser']}")
            summary.append(f"  {row['assigned_tier']} → {row['predicted_tier']}")
            summary.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
            summary.append(f"  💡 {row['recommendation']}")
    else:
        summary.append("  None identified.")

    summary.append("")
    summary.append("-" * 70)
    summary.append("LOW SEVERITY OUTLIERS (Factor Profile Mismatches)")
    summary.append("-" * 70)

    low_sev = recommendations_df[recommendations_df['severity'] == 'LOW'] if len(recommendations_df) > 0 else pd.DataFrame()
    if len(low_sev) > 0:
        for _, row in low_sev.iterrows():
            summary.append(f"\n{row['purchaser']}")
            summary.append(f"  {row['assigned_tier']} → {row['predicted_tier']}")
            summary.append(f"  Win rate: {row['win_rate']*100:.0f}%, Quotes: {row['n_quotes']}")
    else:
        summary.append("  None identified.")

    summary_text = "\n".join(summary)

    with open(f"{output_dir}/_summary.txt", 'w') as f:
        f.write(summary_text)

    print(f"Saved: {output_dir}/_summary.txt")

    return recommendations_df, summary_text


def plot_outliers(df_top, outliers, factor_cols, output_dir):
    """Visualize outliers and tier distributions."""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZATIONS")
    print("=" * 70)

    # 1. Confusion matrix
    tier_order = sorted(df_top['TIER'].unique())
    cm = confusion_matrix(df_top['TIER'], df_top['PREDICTED_TIER'], labels=tier_order)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tier_order, yticklabels=tier_order)
    plt.xlabel('Predicted Tier')
    plt.ylabel('Actual Tier')
    plt.title('Sabel Steel: Actual vs Predicted Price Tier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix.png")

    # 2. Factor distributions by tier
    if len(factor_cols) > 0:
        n_factors = len(factor_cols)
        fig, axes = plt.subplots(1, n_factors, figsize=(5*n_factors, 6))
        if n_factors == 1:
            axes = [axes]

        tier_order = df_top['TIER'].value_counts().index.tolist()

        for ax, factor in zip(axes, factor_cols):
            sns.boxplot(data=df_top, x='TIER', y=factor, order=tier_order, ax=ax)
            ax.set_title(f'{factor} by Tier')
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/factors_by_tier.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/factors_by_tier.png")

    # 3. Outlier scatter plot (if 2+ factors)
    if len(factor_cols) >= 2:
        plt.figure(figsize=(12, 8))

        # Get purchaser-level data
        purchaser_factors = df_top.groupby('PURCHASERCOMPANYNAME')[factor_cols].mean()
        purchaser_tiers = df_top.groupby('PURCHASERCOMPANYNAME')['TIER'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )

        # Plot all purchasers
        scatter = plt.scatter(
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
            outlier_factors = purchaser_factors.loc[outlier_names]
            plt.scatter(
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
                    plt.annotate(
                        name.split('.')[0],  # Shorten domain names
                        (purchaser_factors.loc[name, factor_cols[0]],
                         purchaser_factors.loc[name, factor_cols[1]]),
                        fontsize=8,
                        alpha=0.8
                    )

        plt.xlabel(factor_cols[0])
        plt.ylabel(factor_cols[1])
        plt.title('Sabel Steel Purchasers: Factor Space\n(Red X = Potential Outliers)')
        plt.colorbar(scatter, label='Tier')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/outlier_scatter.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/outlier_scatter.png")

    # 4. Win rate by tier mismatch
    plt.figure(figsize=(10, 6))

    match_stats = df_top.groupby(['TIER', 'TIER_MATCH'])['ISWON'].mean().unstack()
    match_stats.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Assigned Tier')
    plt.ylabel('Win Rate')
    plt.title('Win Rate: Matching vs Non-Matching Tier Predictions')
    plt.legend(['Mismatch (outlier)', 'Match'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/winrate_by_match.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/winrate_by_match.png")


def main():
    """Run Sabel Steel price tier outlier analysis."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Sabel data
    df, tier_map = load_sabel_data()

    # Step 2: Run EFA
    factor_scores, loadings, fa, scaler, factor_names = run_efa(df)
    factor_cols = factor_scores.columns.tolist()

    # Save factor loadings
    loadings.to_csv(f"{OUTPUT_DIR}/factor_loadings.csv")

    # Step 3: Train tier predictor
    df_top, rf, le, importance = train_tier_predictor(df, factor_scores)
    accuracy = df_top['TIER_MATCH'].mean()

    # Step 4: Identify outliers
    purchaser_stats, outliers = identify_outliers(df_top, factor_cols)

    # Step 5: Generate report
    recommendations, summary = generate_report(
        purchaser_stats, outliers, importance, accuracy, factor_cols, OUTPUT_DIR
    )

    # Step 6: Visualizations
    plot_outliers(df_top, outliers, factor_cols, OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"\nKey findings:")
    print(f"  - Model accuracy: {accuracy*100:.1f}%")
    print(f"  - Outliers identified: {len(outliers)} purchasers")

    high_sev = len(recommendations[recommendations['severity'] == 'HIGH']) if len(recommendations) > 0 else 0
    if high_sev > 0:
        print(f"  - HIGH severity (review urgently): {high_sev}")

    print(f"\nSee {OUTPUT_DIR}/_summary.txt for detailed recommendations.")


if __name__ == '__main__':
    main()
