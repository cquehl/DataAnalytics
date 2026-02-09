"""
Statistical Tests Module
========================

Functions for ANOVA, Kolmogorov-Smirnov, and other statistical tests.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional


def run_anova(
    df: pd.DataFrame,
    value_col: str,
    group_col: str
) -> dict:
    """
    Perform one-way ANOVA with effect size calculation.

    Parameters:
        df: Input DataFrame
        value_col: Column with values to compare
        group_col: Column with group labels

    Returns:
        Dictionary with F-statistic, p-value, eta-squared, and interpretation
    """
    # Group the data
    groups = [group[value_col].values for name, group in df.groupby(group_col)]

    # Perform ANOVA
    f_stat, p_value = scipy_stats.f_oneway(*groups)

    # Calculate eta-squared (effect size)
    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((df[value_col] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Interpret effect size
    if eta_squared > 0.14:
        effect_size = "Large"
    elif eta_squared > 0.06:
        effect_size = "Medium"
    else:
        effect_size = "Small"

    # Significance markers
    if p_value < 0.001:
        sig_marker = "***"
    elif p_value < 0.01:
        sig_marker = "**"
    elif p_value < 0.05:
        sig_marker = "*"
    else:
        sig_marker = ""

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'sig_marker': sig_marker,
        'n_groups': len(groups),
        'n_total': len(df),
    }


def run_anova_multi(
    df: pd.DataFrame,
    value_cols: list[str],
    group_col: str
) -> pd.DataFrame:
    """
    Run ANOVA for multiple value columns.

    Parameters:
        df: Input DataFrame
        value_cols: List of columns to analyze
        group_col: Column with group labels

    Returns:
        DataFrame with ANOVA results for each value column
    """
    results = []
    for col in value_cols:
        result = run_anova(df, col, group_col)
        result['variable'] = col
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df[['variable', 'f_statistic', 'p_value', 'eta_squared',
                              'effect_size', 'significant', 'sig_marker']]
    return results_df


def run_ks_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> dict:
    """
    Perform Kolmogorov-Smirnov test to compare two distributions.

    Parameters:
        group1: First sample array
        group2: Second sample array

    Returns:
        Dictionary with KS statistic, p-value, and interpretation
    """
    ks_stat, p_value = scipy_stats.ks_2samp(group1, group2)

    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_group1': len(group1),
        'n_group2': len(group2),
        'interpretation': 'Different distributions' if p_value < 0.05 else 'Similar distributions',
    }


def run_multiple_ks_tests(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    baseline_group: str = None
) -> pd.DataFrame:
    """
    Run KS tests comparing each group to baseline.

    Parameters:
        df: Input DataFrame
        value_col: Column with values to compare
        group_col: Column with group labels
        baseline_group: Group to compare against (default: largest group)

    Returns:
        DataFrame with KS test results for each comparison
    """
    groups = df.groupby(group_col)[value_col]

    if baseline_group is None:
        # Use largest group as baseline
        baseline_group = df[group_col].value_counts().index[0]

    baseline = groups.get_group(baseline_group).values

    results = []
    for name, group in groups:
        if name != baseline_group:
            ks_result = run_ks_test(baseline, group.values)
            ks_result['comparison'] = f"{baseline_group} vs {name}"
            results.append(ks_result)

    return pd.DataFrame(results)


def run_tukey_hsd(
    df: pd.DataFrame,
    value_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    Perform Tukey's HSD post-hoc test.

    Parameters:
        df: Input DataFrame
        value_col: Column with values to compare
        group_col: Column with group labels

    Returns:
        DataFrame with pairwise comparisons
    """
    from scipy.stats import tukey_hsd

    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    group_names = list(df[group_col].unique())

    result = tukey_hsd(*groups)

    # Build comparison DataFrame
    comparisons = []
    for i, name1 in enumerate(group_names):
        for j, name2 in enumerate(group_names):
            if i < j:
                p_val = result.pvalue[i, j]
                comparisons.append({
                    'group1': name1,
                    'group2': name2,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                })

    return pd.DataFrame(comparisons)


def calculate_group_means(
    df: pd.DataFrame,
    value_cols: list[str],
    group_col: str
) -> pd.DataFrame:
    """
    Calculate mean values by group.

    Parameters:
        df: Input DataFrame
        value_cols: Columns to calculate means for
        group_col: Column with group labels

    Returns:
        DataFrame with means by group
    """
    means = df.groupby(group_col)[value_cols].mean()
    means['n'] = df.groupby(group_col).size()
    return means.round(3)


def print_anova_results(anova_df: pd.DataFrame) -> None:
    """Print ANOVA results in readable format."""
    print("\n" + "-" * 60)
    print("ANOVA RESULTS")
    print("-" * 60)

    for _, row in anova_df.iterrows():
        print(f"\n{row['variable']}:")
        print(f"  F-statistic: {row['f_statistic']:.2f}")
        print(f"  p-value: {row['p_value']:.2e} {row['sig_marker']}")
        print(f"  Effect size (eta^2): {row['eta_squared']:.3f} ({row['effect_size']})")
