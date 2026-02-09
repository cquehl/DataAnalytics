"""
Exploratory Factor Analysis Module
===================================

Core EFA functions for factorability testing, factor extraction,
and score calculation.
"""

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def check_factorability(scaled_data: np.ndarray, var_names: list[str]) -> dict:
    """
    Test whether data is suitable for factor analysis.

    Performs:
    - Bartlett's Test of Sphericity: Should be significant (p < 0.05)
    - KMO (Kaiser-Meyer-Olkin): Should be > 0.6, ideally > 0.8

    Parameters:
        scaled_data: Standardized data array (n_samples x n_features)
        var_names: List of variable names

    Returns:
        Dictionary with test results and interpretations
    """
    # Bartlett's test
    chi_square, p_value = calculate_bartlett_sphericity(scaled_data)

    # KMO test
    kmo_all, kmo_model = calculate_kmo(scaled_data)

    results = {
        'bartlett_chi_square': chi_square,
        'bartlett_p_value': p_value,
        'bartlett_pass': p_value < 0.05,
        'kmo_overall': kmo_model,
        'kmo_label': config.get_kmo_label(kmo_model),
        'kmo_per_variable': dict(zip(var_names, kmo_all)),
    }

    # Print results
    print("\n" + "=" * 60)
    print("FACTORABILITY TESTS")
    print("=" * 60)

    print(f"\nBartlett's Test of Sphericity:")
    print(f"  Chi-square: {chi_square:,.2f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Result: {'PASS' if results['bartlett_pass'] else 'FAIL'}")

    print(f"\nKaiser-Meyer-Olkin (KMO) Measure:")
    print(f"  Overall KMO: {kmo_model:.3f} ({results['kmo_label']})")

    print(f"\n  Per-variable KMO:")
    for var, kmo in results['kmo_per_variable'].items():
        label = config.get_kmo_label(kmo)
        print(f"    {var}: {kmo:.3f} ({label})")

    return results


def determine_num_factors(scaled_data: np.ndarray, var_names: list[str] = None) -> tuple:
    """
    Determine optimal number of factors using Kaiser criterion.

    Kaiser criterion: Retain factors with eigenvalue > 1

    Parameters:
        scaled_data: Standardized data array
        var_names: Optional variable names (for sizing)

    Returns:
        Tuple of (eigenvalues array, suggested number of factors)
    """
    n_vars = scaled_data.shape[1] if var_names is None else len(var_names)

    # Fit with max possible factors to get all eigenvalues
    fa = FactorAnalyzer(n_factors=n_vars, rotation=None)
    fa.fit(scaled_data)
    eigenvalues, _ = fa.get_eigenvalues()

    # Kaiser criterion
    kaiser_factors = int(sum(eigenvalues > 1))

    print("\n" + "=" * 60)
    print("FACTOR EXTRACTION CRITERIA")
    print("=" * 60)

    print(f"\nEigenvalues:")
    for i, ev in enumerate(eigenvalues, 1):
        marker = " <-- Kaiser cutoff" if i == kaiser_factors and ev > 1 else ""
        print(f"  Factor {i}: {ev:.3f}{marker}")

    print(f"\nKaiser Criterion (eigenvalue > 1): {kaiser_factors} factors")

    # Cumulative variance
    total_var = sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var * 100
    print(f"\nCumulative variance explained:")
    for i in range(min(kaiser_factors + 1, len(eigenvalues))):
        print(f"  {i+1} factor(s): {cum_var[i]:.1f}%")

    return eigenvalues, kaiser_factors


def run_efa(
    scaled_data: np.ndarray,
    var_names: list[str],
    n_factors: int,
    rotation: str = None
) -> dict:
    """
    Run Exploratory Factor Analysis with specified rotation.

    Parameters:
        scaled_data: Standardized data array
        var_names: List of variable names
        n_factors: Number of factors to extract
        rotation: Rotation method. Defaults to config.DEFAULT_ROTATION

    Returns:
        Dictionary with factor_analyzer, loadings, communalities, variance
    """
    if rotation is None:
        rotation = config.DEFAULT_ROTATION

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(scaled_data)

    # Extract results
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
        index=['Variance', 'Proportional_Var', 'Cumulative_Var'],
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )

    print("\n" + "=" * 60)
    print(f"FACTOR ANALYSIS ({n_factors} factors, {rotation} rotation)")
    print("=" * 60)

    print("\nFactor Loadings:")
    print("-" * 50)
    print(loadings.round(3).to_string())

    print("\nCommunalities:")
    print("-" * 50)
    for var in var_names:
        comm = communalities.loc[var, 'Communality']
        status = "LOW" if comm < 0.4 else "OK"
        print(f"  {var}: {comm:.3f} [{status}]")

    print(f"\nTotal variance explained: {variance[2][-1]*100:.1f}%")

    # Factor interpretation
    print("\n" + "-" * 50)
    print(f"FACTOR INTERPRETATION (loadings > {config.LOADING_THRESHOLD})")
    print("-" * 50)

    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > config.LOADING_THRESHOLD][col]
        high_loaders = high_loaders.reindex(high_loaders.abs().sort_values(ascending=False).index)
        if len(high_loaders) > 0:
            print(f"\n{col}:")
            for var, loading in high_loaders.items():
                sign = "+" if loading > 0 else "-"
                print(f"  {sign} {var}: {loading:.2f}")

    return {
        'factor_analyzer': fa,
        'loadings': loadings,
        'communalities': communalities,
        'variance': variance_df,
        'n_factors': n_factors,
        'rotation': rotation,
    }


def calculate_factor_scores(
    fa: FactorAnalyzer,
    scaled_data: np.ndarray,
    valid_indices: pd.Index = None
) -> pd.DataFrame:
    """
    Calculate factor scores for each observation.

    Parameters:
        fa: Fitted FactorAnalyzer object
        scaled_data: Standardized data array
        valid_indices: Optional index to assign to output DataFrame

    Returns:
        DataFrame with factor scores (n_samples x n_factors)
    """
    scores = fa.transform(scaled_data)
    n_factors = scores.shape[1]

    if valid_indices is not None:
        scores_df = pd.DataFrame(
            scores,
            index=valid_indices,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
    else:
        scores_df = pd.DataFrame(
            scores,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )

    print("\n" + "=" * 60)
    print("FACTOR SCORES")
    print("=" * 60)
    print(f"Calculated {n_factors} factor scores for {len(scores_df):,} observations")
    print("\nFactor Score Statistics:")
    print(scores_df.describe().round(3).to_string())

    return scores_df


def interpret_factors(
    loadings: pd.DataFrame,
    threshold: float = None
) -> dict[str, list[tuple[str, float]]]:
    """
    Generate factor interpretations based on high loadings.

    Parameters:
        loadings: Factor loadings DataFrame
        threshold: Minimum absolute loading to consider. Defaults to config.LOADING_THRESHOLD

    Returns:
        Dictionary mapping factor names to list of (variable, loading) tuples
    """
    if threshold is None:
        threshold = config.LOADING_THRESHOLD

    interpretations = {}
    for col in loadings.columns:
        high_loaders = loadings[abs(loadings[col]) > threshold][col]
        high_loaders = high_loaders.reindex(high_loaders.abs().sort_values(ascending=False).index)
        interpretations[col] = [(var, loading) for var, loading in high_loaders.items()]

    return interpretations


def get_factorability_summary(results: dict) -> pd.DataFrame:
    """
    Convert factorability results to a summary DataFrame.

    Parameters:
        results: Output from check_factorability()

    Returns:
        DataFrame with factorability test results
    """
    rows = [
        {'Test': 'Bartlett_Chi_Square', 'Value': results['bartlett_chi_square'], 'Interpretation': ''},
        {'Test': 'Bartlett_p_value', 'Value': results['bartlett_p_value'],
         'Interpretation': 'PASS' if results['bartlett_pass'] else 'FAIL'},
        {'Test': 'KMO_Overall', 'Value': results['kmo_overall'], 'Interpretation': results['kmo_label']},
    ]

    for var, kmo in results['kmo_per_variable'].items():
        rows.append({
            'Test': f'KMO_{var}',
            'Value': kmo,
            'Interpretation': config.get_kmo_label(kmo)
        })

    return pd.DataFrame(rows)
