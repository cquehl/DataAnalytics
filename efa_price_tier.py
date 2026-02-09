#!/usr/bin/env python3
"""
Exploratory Factor Analysis (EFA) for Price Tier Validation
============================================================

Goal: Discover latent factors in order characteristics, then see if they
discriminate between price tiers.

This is a HYBRID APPROACH:
- EFA finds interpretable factors (what patterns exist in order characteristics?)
- Then we see if those factors differ by tier (do tiers have distinct "profiles"?)

Unlike the margin analysis (continuous DV → regression), price tier is categorical,
so we use:
- ANOVA: Do factor scores differ significantly across tiers?
- Visualization: Box plots showing factor distributions by tier
- Effect sizes: How much do tiers differ on each factor?

Workflow:
1. Load & filter data (Sabel + Coosa, Aug 25+)
2. Select order characteristic variables (IVs)
3. Standardize data (z-score)
4. Check factorability (Bartlett's test, KMO)
5. Determine number of factors (Kaiser criterion, scree plot)
6. Run EFA with Varimax rotation
7. Calculate factor scores for each quote
8. Analyze factor scores by price tier (ANOVA, visualizations)
9. Identify which factors discriminate between tiers
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'Data/Better_Data_Set_Feb3.csv'
SUPPLIERS = ['Sabel Steel', 'coosasteel.com']
START_DATE = '2025-08-25'
OUTPUT_DIR = 'outputs/efa_tier_analysis'
DEPENDENT_VARIABLE = 'PRICETIERKEY'  # Categorical DV


def load_and_filter_data():
    """Load data and filter to specified suppliers and date range."""
    print("=" * 70)
    print("STEP 1: LOADING AND FILTERING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    print(f"Raw data: {len(df):,} records")

    # Filter suppliers
    df = df[df['SUPPLIERCOMPANYNAME'].isin(SUPPLIERS)].copy()
    print(f"After supplier filter ({', '.join(SUPPLIERS)}): {len(df):,} records")

    # Parse and filter dates
    df['DATE'] = pd.to_datetime(df['DATEESTIMATECREATED'])
    df = df[df['DATE'] >= pd.Timestamp(START_DATE)].copy()
    print(f"After date filter ({START_DATE}+): {len(df):,} records")

    # Create readable tier labels (Tier_1 = most common, etc.)
    tier_counts = df['PRICETIERKEY'].value_counts()
    tier_map = {t: f"Tier_{i+1}" for i, t in enumerate(tier_counts.index)}
    df['TIER'] = df['PRICETIERKEY'].map(tier_map)

    print(f"\nUnique tiers: {df['TIER'].nunique()}")
    print(f"Unique purchasers: {df['PURCHASERCOMPANYNAME'].nunique()}")

    return df, tier_map


def select_variables():
    """
    Select order characteristic variables for EFA.

    These should be NUMERIC variables that describe the ORDER,
    not the outcome (tier, margin, win/loss).
    """
    selected_vars = [
        'LINEITEMUNITPRICE',        # Price per unit
        'LINEITEMTOTALPRICE',       # Total line item price
        'ESTIMATETOTALPRICE',       # Total estimate value
        'ESTIMATETOTALQUANTITY',    # Total quantity on estimate
        'QTYPERLINEITEM',           # Quantity per line item
        'MATERIALUNITWEIGHT',       # Weight per unit
        'MATERIALCENTWEIGHTCOST',   # Cost per hundredweight
    ]
    return selected_vars


def standardize_data(df, columns):
    """Standardize selected columns (z-score normalization)."""
    print("\n" + "=" * 70)
    print("STEP 3: STANDARDIZING DATA")
    print("=" * 70)

    # Get valid rows (no NaN in selected columns)
    data = df[columns].dropna()
    valid_indices = df[columns].dropna().index

    print(f"Records with complete data: {len(data):,}")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_array, columns=columns, index=valid_indices)

    print(f"Standardization complete (mean=0, std=1 for each variable)")

    return scaled_array, scaled_df, valid_indices


def check_factorability(scaled_data, var_names):
    """Test whether data is suitable for factor analysis."""
    print("\n" + "=" * 70)
    print("STEP 4: CHECKING FACTORABILITY")
    print("=" * 70)

    # Bartlett's test
    chi_square, p_value = calculate_bartlett_sphericity(scaled_data)

    # KMO test
    kmo_all, kmo_model = calculate_kmo(scaled_data)

    print(f"\nBartlett's Test of Sphericity:")
    print(f"  Chi-square: {chi_square:,.2f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Result: {'PASS' if p_value < 0.05 else 'FAIL'} - ", end="")
    print("Data has correlations suitable for factoring" if p_value < 0.05 else "Variables may be uncorrelated")

    print(f"\nKaiser-Meyer-Olkin (KMO) Measure:")
    print(f"  Overall KMO: {kmo_model:.3f}")

    kmo_labels = {
        0.9: "Marvelous", 0.8: "Meritorious", 0.7: "Middling",
        0.6: "Mediocre", 0.5: "Miserable", 0.0: "Unacceptable"
    }
    kmo_label = next(label for threshold, label in sorted(kmo_labels.items(), reverse=True)
                     if kmo_model >= threshold)
    print(f"  Interpretation: {kmo_label}")

    print(f"\n  Per-variable KMO:")
    for var, kmo in zip(var_names, kmo_all):
        print(f"    {var}: {kmo:.3f}")

    return {
        'bartlett_chi_square': chi_square,
        'bartlett_p_value': p_value,
        'kmo_overall': kmo_model,
        'kmo_per_variable': dict(zip(var_names, kmo_all))
    }


def determine_num_factors(scaled_data, var_names, output_dir):
    """Determine optimal number of factors using Kaiser criterion and scree plot."""
    print("\n" + "=" * 70)
    print("STEP 5: DETERMINING NUMBER OF FACTORS")
    print("=" * 70)

    # Fit with max factors to get eigenvalues
    fa = FactorAnalyzer(n_factors=len(var_names), rotation=None)
    fa.fit(scaled_data)
    eigenvalues, _ = fa.get_eigenvalues()

    # Kaiser criterion
    kaiser_factors = sum(eigenvalues > 1)

    print(f"\nEigenvalues:")
    for i, ev in enumerate(eigenvalues, 1):
        marker = " ← Kaiser cutoff" if ev > 1 and (i == kaiser_factors or eigenvalues[i] <= 1 if i < len(eigenvalues) else True) else ""
        print(f"  Factor {i}: {ev:.3f}{marker}")

    print(f"\nKaiser Criterion: {kaiser_factors} factors (eigenvalue > 1)")

    # Cumulative variance
    total_var = sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var * 100
    print(f"\nCumulative variance explained:")
    for i in range(min(kaiser_factors + 1, len(eigenvalues))):
        print(f"  {i+1} factor(s): {cum_var[i]:.1f}%")

    # Scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
    plt.xlabel('Factor Number', fontsize=11)
    plt.ylabel('Eigenvalue', fontsize=11)
    plt.title('Scree Plot - Finding the "Elbow"', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scree_plot.png", dpi=150)
    plt.close()
    print(f"\nScree plot saved to: {output_dir}/scree_plot.png")

    return eigenvalues, kaiser_factors


def run_factor_analysis(scaled_data, var_names, n_factors, output_dir):
    """Run EFA with Varimax rotation."""
    print("\n" + "=" * 70)
    print(f"STEP 6: FACTOR ANALYSIS ({n_factors} factors, Varimax rotation)")
    print("=" * 70)

    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_data)

    # Get results
    loadings = pd.DataFrame(
        fa.loadings_,
        index=var_names,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    communalities = pd.DataFrame(
        fa.get_communalities(),
        index=var_names,
        columns=['Communality']
    )

    variance = fa.get_factor_variance()

    print("\nFactor Loadings (values > 0.5 indicate strong association):")
    print("-" * 50)
    print(loadings.round(3).to_string())

    print("\nCommunalities (variance explained per variable):")
    print("-" * 50)
    for var, comm in zip(var_names, fa.get_communalities()):
        status = "LOW" if comm < 0.4 else "OK"
        print(f"  {var}: {comm:.3f} [{status}]")

    print(f"\nTotal variance explained: {variance[2][-1]*100:.1f}%")

    # Factor loadings heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Factor Loadings (Varimax Rotation)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_loadings.png", dpi=150)
    plt.close()
    print(f"\nFactor loadings plot saved to: {output_dir}/factor_loadings.png")

    # Interpret factors
    print("\n" + "-" * 50)
    print("FACTOR INTERPRETATION (based on loadings > 0.5)")
    print("-" * 50)
    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > 0.5][col].sort_values(ascending=False)
        if len(high_loaders) > 0:
            print(f"\n{col}:")
            for var, loading in high_loaders.items():
                sign = "+" if loading > 0 else "-"
                print(f"  {sign} {var}: {loading:.2f}")

    return {
        'factor_analyzer': fa,
        'loadings': loadings,
        'communalities': communalities,
        'variance': variance
    }


def calculate_factor_scores(fa, scaled_data, n_factors, valid_indices):
    """Calculate factor scores for each observation."""
    print("\n" + "=" * 70)
    print("STEP 7: CALCULATING FACTOR SCORES")
    print("=" * 70)

    scores = fa.transform(scaled_data)
    scores_df = pd.DataFrame(
        scores,
        index=valid_indices,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    print(f"Calculated factor scores for {len(scores_df):,} quotes")
    print("\nFactor score statistics:")
    print(scores_df.describe().round(3).to_string())

    return scores_df


def analyze_factors_by_tier(df, factor_scores, output_dir, top_n_tiers=5):
    """
    Analyze how factor scores differ across price tiers.

    This is the key step: Do our latent factors discriminate between tiers?
    """
    print("\n" + "=" * 70)
    print("STEP 8: ANALYZING FACTORS BY PRICE TIER")
    print("=" * 70)

    # Merge factor scores with tier info
    analysis_df = df.loc[factor_scores.index, ['TIER', 'ISWON', 'PURCHASERCOMPANYNAME']].copy()
    analysis_df = analysis_df.join(factor_scores)

    # Focus on top N tiers for cleaner analysis
    top_tiers = df['TIER'].value_counts().head(top_n_tiers).index.tolist()
    analysis_df = analysis_df[analysis_df['TIER'].isin(top_tiers)]

    print(f"\nAnalyzing top {top_n_tiers} tiers ({len(analysis_df):,} quotes)")

    factor_cols = [col for col in factor_scores.columns]

    # ANOVA for each factor
    print("\n" + "-" * 50)
    print("ANOVA: Do factor scores differ significantly by tier?")
    print("-" * 50)

    anova_results = []
    for factor in factor_cols:
        groups = [group[factor].values for name, group in analysis_df.groupby('TIER')]
        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - analysis_df[factor].mean())**2 for g in groups)
        ss_total = sum((analysis_df[factor] - analysis_df[factor].mean())**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        effect_size = "Large" if eta_squared > 0.14 else "Medium" if eta_squared > 0.06 else "Small"
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        anova_results.append({
            'Factor': factor,
            'F_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        })

        print(f"\n{factor}:")
        print(f"  F-statistic: {f_stat:.2f}")
        print(f"  p-value: {p_value:.2e} {sig}")
        print(f"  Effect size (η²): {eta_squared:.3f} ({effect_size})")

    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(f"{output_dir}/anova_results.csv", index=False)
    print(f"\nANOVA results saved to: {output_dir}/anova_results.csv")

    # Mean factor scores by tier
    print("\n" + "-" * 50)
    print("MEAN FACTOR SCORES BY TIER")
    print("-" * 50)

    tier_means = analysis_df.groupby('TIER')[factor_cols].mean().round(3)
    tier_means['n_quotes'] = analysis_df.groupby('TIER').size()
    tier_means['win_rate'] = analysis_df.groupby('TIER')['ISWON'].mean().round(3)
    tier_means = tier_means.sort_values('n_quotes', ascending=False)
    print(tier_means.to_string())

    tier_means.to_csv(f"{output_dir}/tier_factor_profiles.csv")
    print(f"\nTier profiles saved to: {output_dir}/tier_factor_profiles.csv")

    # Box plots
    fig, axes = plt.subplots(1, len(factor_cols), figsize=(5*len(factor_cols), 6))
    if len(factor_cols) == 1:
        axes = [axes]

    for ax, factor in zip(axes, factor_cols):
        # Order tiers by frequency
        tier_order = analysis_df['TIER'].value_counts().index.tolist()
        sns.boxplot(data=analysis_df, x='TIER', y=factor, order=tier_order, ax=ax)
        ax.set_title(f'{factor} by Tier')
        ax.set_xlabel('Price Tier')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_boxplots_by_tier.png", dpi=150)
    plt.close()
    print(f"\nBox plots saved to: {output_dir}/factor_boxplots_by_tier.png")

    return analysis_df, anova_df, tier_means


def generate_interpretation(loadings, anova_df, tier_means, output_dir):
    """Generate a human-readable interpretation of results."""
    print("\n" + "=" * 70)
    print("STEP 9: INTERPRETATION")
    print("=" * 70)

    interpretation = []
    interpretation.append("=" * 70)
    interpretation.append("EFA PRICE TIER ANALYSIS - INTERPRETATION")
    interpretation.append("=" * 70)
    interpretation.append("")

    # Factor descriptions
    interpretation.append("WHAT ARE THE FACTORS?")
    interpretation.append("-" * 50)

    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > 0.5][col].sort_values(ascending=False)
        if len(high_loaders) > 0:
            interpretation.append(f"\n{col}:")
            for var, loading in high_loaders.items():
                sign = "higher" if loading > 0 else "lower"
                interpretation.append(f"  - {sign} {var}")

    interpretation.append("")
    interpretation.append("")

    # Which factors discriminate tiers?
    interpretation.append("WHICH FACTORS DISCRIMINATE BETWEEN TIERS?")
    interpretation.append("-" * 50)

    sig_factors = anova_df[anova_df['significant']]
    if len(sig_factors) > 0:
        for _, row in sig_factors.iterrows():
            interpretation.append(f"\n{row['Factor']}: YES (p={row['p_value']:.2e}, η²={row['eta_squared']:.3f})")
            interpretation.append(f"  Effect size: {row['effect_size']}")

    non_sig = anova_df[~anova_df['significant']]
    if len(non_sig) > 0:
        interpretation.append(f"\nNot significant: {', '.join(non_sig['Factor'].tolist())}")

    interpretation.append("")
    interpretation.append("")

    # Tier profiles
    interpretation.append("TIER PROFILES (what makes each tier different)")
    interpretation.append("-" * 50)

    factor_cols = [col for col in tier_means.columns if col.startswith('Factor')]
    for tier in tier_means.index:
        row = tier_means.loc[tier]
        interpretation.append(f"\n{tier} ({int(row['n_quotes'])} quotes, {row['win_rate']*100:.0f}% win rate):")
        for factor in factor_cols:
            val = row[factor]
            if abs(val) > 0.3:
                direction = "HIGH" if val > 0 else "LOW"
                interpretation.append(f"  - {factor}: {direction} ({val:+.2f})")

    interpretation.append("")
    interpretation.append("")
    interpretation.append("=" * 70)
    interpretation.append("HOW TO USE THESE FINDINGS")
    interpretation.append("=" * 70)
    interpretation.append("""
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

    # Save interpretation
    interpretation_text = "\n".join(interpretation)
    print(interpretation_text)

    with open(f"{output_dir}/interpretation.txt", 'w') as f:
        f.write(interpretation_text)
    print(f"\nInterpretation saved to: {output_dir}/interpretation.txt")

    return interpretation_text


def main():
    """Run the full EFA price tier analysis."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load and filter data
    df, tier_map = load_and_filter_data()

    # Step 2: Select variables
    print("\n" + "=" * 70)
    print("STEP 2: SELECTING VARIABLES")
    print("=" * 70)
    selected_vars = select_variables()
    print(f"Selected {len(selected_vars)} order characteristic variables:")
    for var in selected_vars:
        print(f"  - {var}")

    # Step 3: Standardize
    scaled_data, scaled_df, valid_indices = standardize_data(df, selected_vars)

    # Step 4: Check factorability
    factorability = check_factorability(scaled_data, selected_vars)

    # Step 5: Determine number of factors
    eigenvalues, suggested_factors = determine_num_factors(scaled_data, selected_vars, OUTPUT_DIR)

    # Step 6: Run factor analysis
    n_factors = suggested_factors
    results = run_factor_analysis(scaled_data, selected_vars, n_factors, OUTPUT_DIR)

    # Step 7: Calculate factor scores
    factor_scores = calculate_factor_scores(
        results['factor_analyzer'], scaled_data, n_factors, valid_indices
    )

    # Step 8: Analyze by tier
    analysis_df, anova_df, tier_means = analyze_factors_by_tier(
        df, factor_scores, OUTPUT_DIR, top_n_tiers=5
    )

    # Step 9: Generate interpretation
    generate_interpretation(results['loadings'], anova_df, tier_means, OUTPUT_DIR)

    # Save factor scores for use in hybrid model
    factor_scores.to_csv(f"{OUTPUT_DIR}/factor_scores.csv")
    print(f"\nFactor scores saved to: {OUTPUT_DIR}/factor_scores.csv")
    print("(Use these as inputs to the Random Forest model for hybrid approach)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
