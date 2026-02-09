"""
Global Configuration for EFA Analysis Framework
================================================

Central location for default parameters used across all analysis scripts.
Override these in individual test scripts as needed.
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DEFAULT_DATA_FILE = 'Data/Decent_Data_Set_Feb3.csv'

DEFAULT_SUPPLIERS = ['Sabel Steel', 'coosasteel.com']
DEFAULT_START_DATE = '2025-08-25'

# =============================================================================
# EFA CONFIGURATION
# =============================================================================
# Standard 7-variable feature set for factor analysis
DEFAULT_EFA_FEATURES = [
    'LINEITEMUNITPRICE',        # Price per unit
    'LINEITEMTOTALPRICE',       # Total line item price
    'ESTIMATETOTALPRICE',       # Total estimate value
    'ESTIMATETOTALQUANTITY',    # Total quantity on estimate
    'QTYPERLINEITEM',           # Quantity per line item
    'MATERIALUNITWEIGHT',       # Weight per unit
    'MATERIALCENTWEIGHTCOST',   # Cost per hundredweight (material cost rate)
]

# EFA parameters
DEFAULT_ROTATION = 'varimax'
LOADING_THRESHOLD = 0.5  # Threshold for "high" factor loadings

# =============================================================================
# RANDOM FOREST CONFIGURATION
# =============================================================================
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_leaf': 10,
    'random_state': 42,
}

RF_CV_FOLDS = 5  # Cross-validation folds

# Number of top tiers to include in analysis
DEFAULT_N_TOP_TIERS = 5

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
DEFAULT_OUTPUT_BASE = 'outputs'
DEFAULT_DPI = 150

# =============================================================================
# KMO INTERPRETATION LABELS
# =============================================================================
KMO_THRESHOLDS = {
    0.9: "Marvelous",
    0.8: "Meritorious",
    0.7: "Middling",
    0.6: "Mediocre",
    0.5: "Miserable",
    0.0: "Unacceptable",
}


def get_kmo_label(kmo_value: float) -> str:
    """Return human-readable KMO interpretation."""
    for threshold, label in sorted(KMO_THRESHOLDS.items(), reverse=True):
        if kmo_value >= threshold:
            return label
    return "Unacceptable"
