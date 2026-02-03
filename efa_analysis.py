"""
Exploratory Factor Analysis (EFA) for Manufacturing Quote Data
Goal: Discover latent factors in order characteristics, then use them to predict MARGIN

Workflow:
1. Select & Standardize IVs (Independent Variables - order characteristics)
2. Check Correlation Matrix & Factorability
3. Determine Number of Factors (Kaiser Criterion & Scree Plot)
4. Extract Factors with Varimax Rotation
5. Calculate Factor Scores for each observation
6. Regress Factor Scores on DV (MARGINACTUAL) to see which factors predict margin
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')  # Suppress numerical warnings for cleaner output


# =============================================================================
# CONFIGURATION
# =============================================================================
DEPENDENT_VARIABLE = 'MARGINACTUAL'  # What we're trying to predict


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load the CSV and return the full dataframe."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def select_variables_for_efa(df: pd.DataFrame) -> list[str]:
    """
    Select which numeric IVs to include in factor analysis.
    These should be ORDER CHARACTERISTICS that might predict margin.
    Do NOT include margin variables here - margin is the DV we're predicting.
    """
    # With the larger dataset (1000 records, varying quantities),
    # we can include more variables without singularity issues
    selected_vars = [
        'LINEITEMUNITPRICE',        # Price per unit
        'LINEITEMTOTALPRICE',       # Total line item price
        'ESTIMATETOTALPRICE',       # Total estimate value
        'ESTIMATETOTALQUANTITY',    # Total quantity on estimate
        'QTYPERLINEITEM',           # Quantity per line item
        'MATERIALUNITWEIGHT',       # Weight per unit
        'MATERIALCENTWEIGHTCOST',   # Cost per hundredweight (material cost rate)
    ]
    return selected_vars


def standardize_data(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Standardize the selected columns (z-score normalization).
    Returns both the scaled array and a DataFrame version.
    """
    data = df[columns].dropna()
    print(f"After dropping NaN: {len(data)} records")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_array, columns=columns)

    print("\nStandardization complete:")
    print(f"  Mean of each variable: ~0 (verified: {scaled_array.mean(axis=0).round(10)})")
    print(f"  Std of each variable:  ~1 (verified: {scaled_array.std(axis=0).round(2)})")

    return scaled_array, scaled_df


def check_factorability(scaled_data: np.ndarray, var_names: list[str]) -> dict:
    """
    Test whether the data is suitable for factor analysis.
    - Bartlett's Test: Should be significant (p < 0.05)
    - KMO (Kaiser-Meyer-Olkin): Should be > 0.6, ideally > 0.8
    """
    # Bartlett's test of sphericity
    chi_square, p_value = calculate_bartlett_sphericity(scaled_data)

    # KMO test
    kmo_all, kmo_model = calculate_kmo(scaled_data)

    results = {
        'bartlett_chi_square': chi_square,
        'bartlett_p_value': p_value,
        'kmo_per_variable': dict(zip(var_names, kmo_all)),
        'kmo_overall': kmo_model
    }

    print("\n" + "="*50)
    print("FACTORABILITY TESTS")
    print("="*50)
    print(f"\nBartlett's Test of Sphericity:")
    print(f"  Chi-square: {chi_square:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Interpretation: {'PASS - Data is suitable' if p_value < 0.05 else 'FAIL - Variables may be uncorrelated'}")

    print(f"\nKaiser-Meyer-Olkin (KMO) Measure:")
    print(f"  Overall KMO: {kmo_model:.3f}")
    kmo_interpretation = (
        "Excellent" if kmo_model >= 0.9 else
        "Good" if kmo_model >= 0.8 else
        "Acceptable" if kmo_model >= 0.7 else
        "Mediocre" if kmo_model >= 0.6 else
        "Poor - Consider removing variables"
    )
    print(f"  Interpretation: {kmo_interpretation}")

    return results


def plot_correlation_matrix(scaled_df: pd.DataFrame, save_path: str = None):
    """Visualize the correlation matrix as a heatmap."""
    corr_matrix = scaled_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Selected Variables')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nCorrelation matrix saved to: {save_path}")
    plt.close()

    return corr_matrix


def determine_num_factors(scaled_data: np.ndarray, var_names: list[str], save_path: str = None):
    """
    Use Kaiser Criterion and Scree Plot to determine optimal number of factors.
    """
    # Fit with max possible factors to get all eigenvalues
    fa_initial = FactorAnalyzer(n_factors=len(var_names), rotation=None)
    fa_initial.fit(scaled_data)

    eigenvalues, _ = fa_initial.get_eigenvalues()

    print("\n" + "="*50)
    print("FACTOR EXTRACTION CRITERIA")
    print("="*50)

    # Kaiser Criterion
    kaiser_factors = sum(eigenvalues > 1)
    print(f"\nEigenvalues:")
    for i, ev in enumerate(eigenvalues, 1):
        marker = " <-- Kaiser cutoff" if i == kaiser_factors and ev > 1 else ""
        print(f"  Factor {i}: {ev:.3f}{marker}")

    print(f"\nKaiser Criterion (eigenvalue > 1): Suggests {kaiser_factors} factors")

    # Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot - Finding the "Elbow"')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(eigenvalues) + 1))

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Scree plot saved to: {save_path}")
    plt.close()

    return eigenvalues, kaiser_factors


def run_factor_analysis(scaled_data: np.ndarray, var_names: list[str],
                        n_factors: int, rotation: str = 'varimax') -> dict:
    """
    Run the actual factor analysis with specified number of factors and rotation.
    """
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
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
    variance_df = pd.DataFrame(
        variance,
        index=['Variance', 'Proportional Var', 'Cumulative Var'],
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    print("\n" + "="*50)
    print(f"FACTOR ANALYSIS RESULTS ({n_factors} factors, {rotation} rotation)")
    print("="*50)

    print("\nFactor Loadings:")
    print("-" * 40)
    print(loadings.round(3).to_string())

    print("\nCommunalities (variance explained per variable):")
    print("-" * 40)
    print(communalities.round(3).to_string())
    low_communality = communalities[communalities['Communality'] < 0.4]
    if len(low_communality) > 0:
        print(f"\n  Warning: Variables with low communality (<0.4):")
        for var in low_communality.index:
            print(f"    - {var}: {communalities.loc[var, 'Communality']:.3f}")

    print("\nVariance Explained:")
    print("-" * 40)
    print(variance_df.round(3).to_string())
    print(f"\nTotal variance explained: {variance[2][-1]*100:.1f}%")

    return {
        'factor_analyzer': fa,
        'loadings': loadings,
        'communalities': communalities,
        'variance': variance_df
    }


def plot_factor_loadings(loadings: pd.DataFrame, save_path: str = None):
    """Visualize factor loadings as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Factor Loadings (Varimax Rotation)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Factor loadings plot saved to: {save_path}")
    plt.close()


def calculate_factor_scores(fa: FactorAnalyzer, scaled_data: np.ndarray,
                            n_factors: int) -> pd.DataFrame:
    """
    Calculate factor scores for each observation.
    These become the new IVs for regression.
    """
    scores = fa.transform(scaled_data)
    scores_df = pd.DataFrame(
        scores,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    print("\n" + "="*50)
    print("FACTOR SCORES")
    print("="*50)
    print(f"\nCalculated {n_factors} factor scores for {len(scores_df)} observations")
    print("\nFactor Score Statistics:")
    print(scores_df.describe().round(3).to_string())

    return scores_df


def regress_factors_on_dv(factor_scores: pd.DataFrame, dv_values: pd.Series,
                          factor_names: list[str] = None) -> dict:
    """
    Run OLS regression: DV ~ Factor_1 + Factor_2 + ...
    This tells us which latent factors predict the dependent variable.
    """
    # Prepare data
    X = factor_scores.copy()
    if factor_names:
        X.columns = factor_names
    X = sm.add_constant(X)  # Add intercept
    y = dv_values.values

    # Fit regression
    model = sm.OLS(y, X).fit()

    print("\n" + "="*50)
    print(f"REGRESSION: {DEPENDENT_VARIABLE} ~ Factors")
    print("="*50)
    print(model.summary())

    return {
        'model': model,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'coefficients': model.params,
        'p_values': model.pvalues
    }


def plot_regression_coefficients(regression_results: dict, save_path: str = None):
    """
    Visualize regression coefficients with significance indicators.

    Creates a horizontal bar chart showing:
    - Beta coefficients for each factor (excluding intercept)
    - Color-coding by statistical significance (p < 0.05)
    - Error bars showing 95% confidence intervals
    """
    model = regression_results['model']
    coeffs = regression_results['coefficients']
    p_values = regression_results['p_values']

    # Get confidence intervals (returns DataFrame with columns 0 and 1 for lower/upper)
    conf_int = model.conf_int()

    # Exclude the intercept (const) - we only want factor coefficients
    factor_names = [name for name in coeffs.index if name != 'const']
    factor_coeffs = coeffs[factor_names]
    factor_pvals = p_values[factor_names]
    factor_ci = conf_int.loc[factor_names]

    # Calculate error bar sizes (distance from coefficient to CI bounds)
    errors = np.array([
        factor_coeffs.values - factor_ci[0].values,  # lower error
        factor_ci[1].values - factor_coeffs.values   # upper error
    ])

    # Assign colors based on significance
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in factor_pvals]

    plt.figure(figsize=(10, 6))

    # Create horizontal bar chart
    bars = plt.barh(factor_names, factor_coeffs, xerr=errors, color=colors,
                    edgecolor='black', linewidth=0.5, capsize=4)

    # Add reference line at zero
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Labels and title
    plt.xlabel('Regression Coefficient (β)', fontsize=11)
    plt.ylabel('Factor', fontsize=11)
    plt.title(f'Factor Effects on {DEPENDENT_VARIABLE}\n(green = p < 0.05, gray = not significant)',
              fontsize=12)

    # Add coefficient values as text labels
    for bar, coef, pval in zip(bars, factor_coeffs, factor_pvals):
        x_pos = coef + (0.02 if coef >= 0 else -0.02)
        ha = 'left' if coef >= 0 else 'right'
        sig_marker = '*' if pval < 0.05 else ''
        plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f'{coef:.3f}{sig_marker}', va='center', ha=ha, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Regression coefficients plot saved to: {save_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "Data/NerdClustersSharedData_2026-01-12-1020.csv"
    OUTPUT_DIR = "outputs"

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    print("STEP 1: Loading Data")
    print("-" * 50)
    df = load_and_prepare_data(DATA_PATH)

    # Step 2: Select variables (this is where your choice matters!)
    print("\n\nSTEP 2: Selecting Variables for Analysis")
    print("-" * 50)
    selected_vars = select_variables_for_efa(df)
    print(f"Selected {len(selected_vars)} variables: {selected_vars}")

    # Step 3: Standardize
    print("\n\nSTEP 3: Standardizing Data")
    print("-" * 50)
    scaled_data, scaled_df = standardize_data(df, selected_vars)

    # Step 4: Check if data is suitable for EFA
    print("\n\nSTEP 4: Checking Factorability")
    factorability = check_factorability(scaled_data, selected_vars)

    # Step 5: Visualize correlations
    print("\n\nSTEP 5: Correlation Matrix")
    print("-" * 50)
    corr_matrix = plot_correlation_matrix(scaled_df, f"{OUTPUT_DIR}/correlation_matrix.png")

    # Step 6: Determine number of factors
    print("\n\nSTEP 6: Determining Number of Factors")
    eigenvalues, suggested_factors = determine_num_factors(
        scaled_data, selected_vars, f"{OUTPUT_DIR}/scree_plot.png"
    )

    # Step 7: Run factor analysis
    print("\n\nSTEP 7: Running Factor Analysis")
    n_factors = suggested_factors  # Use Kaiser criterion suggestion
    results = run_factor_analysis(scaled_data, selected_vars, n_factors, rotation='varimax')

    # Step 8: Visualize loadings
    print("\n\nSTEP 8: Visualizing Factor Loadings")
    plot_factor_loadings(results['loadings'], f"{OUTPUT_DIR}/factor_loadings.png")

    # Step 9: Calculate factor scores
    print("\n\nSTEP 9: Calculating Factor Scores")
    factor_scores = calculate_factor_scores(
        results['factor_analyzer'], scaled_data, n_factors
    )

    # Step 10: Prepare DV (margin) - need to align indices after dropna
    print("\n\nSTEP 10: Preparing Dependent Variable")
    print("-" * 50)

    # Get indices where IVs are valid (matches what standardize_data used)
    iv_valid_indices = df[selected_vars].dropna().index

    # Check which of those rows ALSO have valid DV values
    dv_valid_mask = df.loc[iv_valid_indices, DEPENDENT_VARIABLE].notna().values

    # Filter factor_scores to only include rows with valid DV
    factor_scores_filtered = factor_scores[dv_valid_mask].reset_index(drop=True)
    dv_data = df.loc[iv_valid_indices, DEPENDENT_VARIABLE][dv_valid_mask].reset_index(drop=True)

    n_dropped = len(iv_valid_indices) - len(dv_data)
    if n_dropped > 0:
        print(f"  Note: Dropped {n_dropped} rows with NaN in {DEPENDENT_VARIABLE}")
    print(f"  Final sample size for regression: {len(dv_data)}")

    print(f"\nDV: {DEPENDENT_VARIABLE}")
    print(f"  Mean: {dv_data.mean():.4f}")
    print(f"  Std:  {dv_data.std():.4f}")
    print(f"  Min:  {dv_data.min():.4f}")
    print(f"  Max:  {dv_data.max():.4f}")

    # Step 11: Regression - which factors predict margin?
    print("\n\nSTEP 11: Regressing Factors on Margin")
    regression_results = regress_factors_on_dv(factor_scores_filtered, dv_data)

    # Step 12: Visualize regression results
    print("\n\nSTEP 12: Visualizing Regression Coefficients")
    plot_regression_coefficients(regression_results, f"{OUTPUT_DIR}/regression_coefficients.png")

    # Save all results
    results['loadings'].to_csv(f"{OUTPUT_DIR}/factor_loadings.csv")
    results['communalities'].to_csv(f"{OUTPUT_DIR}/communalities.csv")
    factor_scores.to_csv(f"{OUTPUT_DIR}/factor_scores.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/ directory")

    print("\n" + "="*50)
    print("INTERPRETATION GUIDE")
    print("="*50)
    print("""
1. FACTOR LOADINGS: Which variables group together?
   - Look for loadings > 0.5 or < -0.5
   - Name each factor based on what the variables represent

2. REGRESSION RESULTS: Which factors predict margin?
   - Significant p-values (< 0.05) indicate predictive factors
   - Positive coefficients = higher factor score → higher margin
   - R² tells you total variance explained

3. NEXT STEPS:
   - Add 'PRICING_TIER' or 'CUSTOMER_TIER' as a new DV
   - Try different factor solutions (2, 3, or 4 factors)
   - Consider oblique rotation (promax) if factors correlate
    """)
