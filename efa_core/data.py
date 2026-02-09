"""
Data Loading and Preprocessing Module
======================================

Functions for loading, filtering, and standardizing data for analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union

from . import config


def load_csv(filepath: str = None) -> pd.DataFrame:
    """
    Load CSV file with basic validation.

    Parameters:
        filepath: Path to CSV file. Defaults to config.DEFAULT_DATA_FILE

    Returns:
        DataFrame with loaded data
    """
    if filepath is None:
        filepath = config.DEFAULT_DATA_FILE

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns from {filepath}")
    return df


def filter_by_suppliers(
    df: pd.DataFrame,
    suppliers: list[str] = None,
    supplier_col: str = 'SUPPLIERCOMPANYNAME'
) -> pd.DataFrame:
    """
    Filter DataFrame to specified suppliers.

    Parameters:
        df: Input DataFrame
        suppliers: List of supplier names to include. Defaults to config.DEFAULT_SUPPLIERS
        supplier_col: Column containing supplier names

    Returns:
        Filtered DataFrame
    """
    if suppliers is None:
        suppliers = config.DEFAULT_SUPPLIERS

    df_filtered = df[df[supplier_col].isin(suppliers)].copy()
    print(f"After supplier filter ({', '.join(suppliers)}): {len(df_filtered):,} records")
    return df_filtered


def filter_by_date(
    df: pd.DataFrame,
    start_date: str = None,
    date_col: str = 'DATEESTIMATECREATED'
) -> pd.DataFrame:
    """
    Filter DataFrame to records on or after start_date.

    Parameters:
        df: Input DataFrame
        start_date: Filter from this date (YYYY-MM-DD). Defaults to config.DEFAULT_START_DATE
        date_col: Column containing dates

    Returns:
        Filtered DataFrame with parsed DATE column
    """
    if start_date is None:
        start_date = config.DEFAULT_START_DATE

    df = df.copy()
    df['DATE'] = pd.to_datetime(df[date_col])
    df = df[df['DATE'] >= pd.Timestamp(start_date)]
    print(f"After date filter ({start_date}+): {len(df):,} records")
    return df


def create_tier_labels(
    df: pd.DataFrame,
    tier_col: str = 'PRICETIERKEY',
    output_col: str = 'TIER'
) -> tuple[pd.DataFrame, dict]:
    """
    Create readable tier labels based on frequency.

    Tier_1 = most common tier, Tier_2 = second most common, etc.

    Parameters:
        df: Input DataFrame
        tier_col: Column containing tier keys
        output_col: Name for new tier label column

    Returns:
        Tuple of (DataFrame with new tier column, tier mapping dict)
    """
    df = df.copy()
    tier_counts = df[tier_col].value_counts()
    tier_map = {t: f"Tier_{i+1}" for i, t in enumerate(tier_counts.index)}
    df[output_col] = df[tier_col].map(tier_map)

    print(f"Created tier labels: {df[output_col].nunique()} unique tiers")
    return df, tier_map


def standardize_features(
    df: pd.DataFrame,
    columns: list[str] = None
) -> tuple[np.ndarray, pd.DataFrame, pd.Index, StandardScaler]:
    """
    Z-score normalize selected columns.

    Parameters:
        df: Input DataFrame
        columns: Columns to standardize. Defaults to config.DEFAULT_EFA_FEATURES

    Returns:
        Tuple of (scaled array, scaled DataFrame, valid indices, fitted scaler)
    """
    if columns is None:
        columns = config.DEFAULT_EFA_FEATURES

    # Get rows with complete data
    data = df[columns].dropna()
    valid_indices = data.index

    print(f"Records with complete data: {len(data):,}")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_array, columns=columns, index=valid_indices)

    print(f"Standardization complete (mean≈0, std≈1 for each variable)")

    return scaled_array, scaled_df, valid_indices, scaler


def get_default_efa_features() -> list[str]:
    """
    Return the standard 7-variable list for EFA.

    Returns:
        List of column names for EFA
    """
    return config.DEFAULT_EFA_FEATURES.copy()


def load_and_prepare(
    filepath: str = None,
    suppliers: list[str] = None,
    start_date: str = None,
    create_tiers: bool = True
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
    """
    Convenience function: load, filter, and optionally create tier labels.

    Parameters:
        filepath: Path to CSV file
        suppliers: List of supplier names to filter
        start_date: Filter from this date
        create_tiers: Whether to create tier labels

    Returns:
        DataFrame, or tuple of (DataFrame, tier_map) if create_tiers=True
    """
    df = load_csv(filepath)
    df = filter_by_suppliers(df, suppliers)
    df = filter_by_date(df, start_date)

    if create_tiers:
        return create_tier_labels(df)
    return df
