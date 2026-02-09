# Code Review Report
**Date**: February 9, 2026
**Codebase**: Exploratory Factor Analysis Project
**Total**: 7 Python scripts (~2,900 lines)

---

## Executive Summary

**Strengths**: Well-documented code with excellent docstrings, type hints, and step-by-step console output. Clear statistical methodology.

**Key Issues Found**:
- 1 critical issue (FIXED)
- 2 high-priority issues (for future refactor)
- 3 medium-priority issues
- 3 low-priority issues

---

## Issues Found

### CRITICAL (Fixed)

#### Dependency Enforcement in `price_tier_validation_hybrid.py`
**Status**: FIXED

**Problem**: Script silently degraded when `efa_price_tier.py` hadn't been run first, falling back to raw features without the user realizing the hybrid approach wasn't being used.

**Fix Applied**: Changed to raise `FileNotFoundError` with clear instructions:
```python
if not os.path.exists(FACTOR_SCORES_FILE):
    raise FileNotFoundError(
        f"Factor scores not found at: {FACTOR_SCORES_FILE}\n"
        f"This hybrid analysis REQUIRES factor scores from EFA.\n"
        f"Please run: python efa_price_tier.py\n"
    )
```

---

### HIGH Priority (Future Refactor)

#### 1. Code Duplication

| Duplicated Pattern | Files | Approx Lines |
|--------------------|-------|--------------|
| EFA workflow (standardize → factorability → extract → rotate → score) | efa_analysis.py, efa_price_tier.py, sabel_price_tier_outliers.py | ~250 |
| Data filtering (supplier + date) | 4+ scripts | ~120 |
| Random Forest training | price_tier_validation.py, price_tier_validation_hybrid.py, sabel_price_tier_outliers.py | ~120 |
| Tier label creation | 3+ scripts | ~30 |

**Recommendation**: Extract to shared `efa_utils.py` module with functions:
- `run_efa(data, var_names, rotation='varimax')`
- `load_and_filter_data(filepath, suppliers, start_date)`
- `create_tier_labels(df, tier_column)`
- `train_rf_classifier(X, y, n_estimators=100)`

#### 2. Hardcoded Path Diagrams

**Files**: `plot_path_diagram.py`, `plot_path_diagram_3factor.py`

These files contain hardcoded statistical results that become stale when analysis is re-run:

```python
# plot_path_diagram_3factor.py lines 120-140
draw_arrow((4.1, 7.9), (6.4, factor1_y + 0.4), '0.75', color='#1976D2')
ax.text(13.6, 2.8, 'R² = 0.021', ha='center', va='center')
```

**Recommendation**: Either:
- Auto-generate from CSV outputs (factor_loadings.csv, regression_results.csv)
- Add `# HARDCODED: Last updated YYYY-MM-DD` comments at top of each file

---

### MEDIUM Priority

#### 3. Missing Input Validation

**All data-loading scripts**

No checks for:
- Required columns exist before accessing (`df['SUPPLIERCOMPANYNAME']`)
- Sufficient sample size for cross-validation (n < cv folds)
- Correlation matrix positive definiteness (EFA requirement)

**Example fix**:
```python
required_cols = ['SUPPLIERCOMPANYNAME', 'DATEESTIMATECREATED', 'PRICETIERKEY']
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")
```

#### 4. Limited Statistical Error Handling

**Files**: efa_analysis.py, efa_price_tier.py, sabel_price_tier_outliers.py

No handling for:
- Singular correlation matrices (factorability check passes but extraction fails)
- Empty groups in ANOVA (`scipy.stats.f_oneway` with n=0 groups)
- Factor analysis convergence failures

**Recommendation**: Wrap in try/except with informative messages:
```python
try:
    fa.fit(scaled_data)
except Exception as e:
    raise RuntimeError(f"Factor analysis failed. Check data for multicollinearity: {e}")
```

#### 5. Inconsistent Output Naming

**Directories**:
- `outputs/efa_tier_analysis/`
- `outputs/hybrid_validation/`
- `outputs/sabel_outliers/`
- `outputs (Feb 4)/` (archived with spaces in name)

**Factor naming**:
```python
# efa_price_tier.py - explicit mapping
FACTOR_NAMES = {'Factor_1': 'Unit_Characteristics', ...}

# sabel_price_tier_outliers.py - inferred heuristically
name = 'Unit_Characteristics'  # Based on loading patterns
```

---

### LOW Priority

#### 6. Incomplete Type Hints
Some functions missing return type hints:
- `load_data()` → should be `tuple[pd.DataFrame, pd.DataFrame]`
- `analyze_purchasers()` → should be `tuple[pd.DataFrame, pd.DataFrame]`

#### 7. Missing main() Docstrings
Files without `main()` docstrings:
- sabel_price_tier_outliers.py
- (price_tier_validation_hybrid.py has brief one now)

#### 8. Pandas SettingWithCopyWarning Risk
Some scripts modify DataFrames without explicit `.copy()`:
```python
df = df[df['SUPPLIERCOMPANYNAME'].isin(suppliers)]  # Should add .copy()
df['DATE'] = pd.to_datetime(...)  # May warn
```

---

## What's Working Well

1. **Excellent Documentation**: Every function has clear docstrings with purpose and parameters
2. **Step-by-step Output**: Console output guides users through the analysis with clear section markers
3. **Type Hints**: Consistent use throughout (Python 3.13 compatible)
4. **Configuration at Top**: Easy to change parameters (DATA_FILE, SUPPLIERS, OUTPUT_DIR)
5. **Non-interactive Backend**: `matplotlib.use('Agg')` enables headless execution
6. **Output Directory Pattern**: All scripts follow `os.makedirs(OUTPUT_DIR, exist_ok=True)` per CLAUDE.md

---

## Script Dependency Graph

```
efa_analysis.py (standalone)
    └── plot_path_diagram.py (reads hardcoded values)

efa_price_tier.py (standalone)
    ├── plot_path_diagram_3factor.py (reads hardcoded values)
    └── price_tier_validation_hybrid.py (REQUIRES factor_scores.csv)

price_tier_validation.py (standalone - raw features only)

sabel_price_tier_outliers.py (standalone - internal EFA)
```

---

## Recommended Refactor Order (Future)

1. **Create `efa_utils.py`** - Extract shared EFA workflow
2. **Create `data_utils.py`** - Extract data loading and filtering
3. **Update scripts** to use shared utilities
4. **Add input validation** to shared loaders
5. **Add tests** for extracted utilities
6. **Auto-generate path diagrams** from CSV outputs
