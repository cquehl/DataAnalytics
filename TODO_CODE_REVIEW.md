# Code Review TODO - PR #2: Modular Framework

> Generated from code review on 2026-02-09
> Branch: `feature/modular-framework-orchestrators`

---

## 🔴 Critical (Fix Before Merge)

### ~~1. Unstable Tier Labeling~~ → DOWNGRADED to Low
- **File**: `efa_core/data.py:107`
- **Original Concern**: Tier labels based on frequency ordering break reproducibility
- **Resolution**: Tier names are being added to the dataset. Once available, use actual tier names instead of frequency-based labels.
- **Business Context**:
  - Tiers group similar customers for standardized pricing
  - Each tier has: base margin level, quantity-based price breaks, product-specific margins
  - Customer tier mobility is a real business scenario (not a bug)
- **Action When Names Arrive**:
  - [ ] Update `create_tier_labels()` to use actual tier name column
  - [ ] Keep count as metadata (useful for sorting, statistical validity)
  - [ ] Add tier stats output (count, avg margin, win rate per tier)
- **Status**: ⏸️ Waiting for data update

---

### 2. ~~Silent Data Corruption via NaN Fill~~ ✅ FIXED
- **File**: `efa_core/models.py:46`
- **Problem**: `fillna(0)` introduces bias for price/quantity features
- **Fix Applied**:
  - `train_rf_classifier()`: Now drops rows with missing values, logs count dropped
  - `predict_with_model()`: Returns `None` for rows with missing values
  - Added TODO comment re: future operations/services needing separate handling
- **Status**: ✅ Complete

---

### 3. ~~Fragile Import Pattern~~ ✅ FIXED
- **Files**: All `efa_core/*.py` modules + `analyses/*.py` scripts
- **Problem**: `sys.path.insert(0, ...)` breaks if package is pip-installed
- **Fix Applied (Hybrid Approach)**:
  - [x] Moved `config.py` → `efa_core/config.py`
  - [x] Updated `efa_core/__init__.py` to export config
  - [x] Updated 5 efa_core modules to use `from . import config`
  - [x] Updated 8 analyses scripts with try/except hybrid import pattern
  - [x] Created `pyproject.toml` for pip-installability
  - [x] Deleted root `config.py`
- **Now works both ways**:
  - Direct: `python analyses/run_efa.py` (fallback to sys.path)
  - Installed: `pip install -e .` then import from anywhere
- **Status**: ✅ Complete

---

## 🟡 Moderate (Fix Soon)

### 4. Warnings Silenced Globally
- **Files**: `analyses/full_tier_analysis.py:36`, `analyses/quick_validation.py:30`, etc.
- **Problem**: `warnings.filterwarnings('ignore')` hides important warnings
- **Impact**: Convergence issues, deprecation warnings go unnoticed
- **Fix**:
  - [ ] Use context managers for targeted suppression
  - [ ] Or filter only specific warning categories
  ```python
  # Better approach
  with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=ConvergenceWarning)
      # factor analysis code here
  ```
- **Status**: ⬜ Not started

---

### 5. Mixed Return Types in load_and_prepare()
- **File**: `efa_core/data.py:156-180`
- **Problem**: Returns `DataFrame` or `tuple[DataFrame, dict]` based on flag
- **Impact**: Type checkers confused; IDE autocomplete unreliable
- **Fix Options**:
  - [ ] Always return tuple (breaking change)
  - [ ] Split into `load_and_prepare()` and `load_and_prepare_with_tiers()`
  - [ ] Use a result dataclass
- **Status**: ⬜ Not started

---

### 6. Missing Input Validation
- **Files**: Multiple
- **Locations**:
  - `efa_core/data.py:32` - `load_csv()` no file existence check
  - `efa_core/data.py:56` - No validation that `supplier_col` exists
  - `efa_core/data.py:132` - No validation that feature columns exist
- **Fix**:
  - [ ] Add `Path(filepath).exists()` check with helpful error message
  - [ ] Validate column existence before filtering/selecting
  - [ ] Raise `KeyError` with list of available columns
- **Status**: ⬜ Not started

---

### 7. Double Scaling in RF Pipeline
- **File**: `efa_core/models.py:53-54`
- **Problem**: Factor scores already standardized; RF pipeline scales again
- **Context**: `full_outlier_detection.py` flow:
  1. `data.standardize_features()` → scaled data
  2. `efa.calculate_factor_scores()` → factor scores (already standardized)
  3. `models.train_rf_classifier()` → scales AGAIN
- **Fix**:
  - [ ] Add `scale=True` parameter to `train_rf_classifier()`
  - [ ] Document when scaling is already done
- **Status**: ⬜ Not started

---

### 8. Date Column Side Effect
- **File**: `efa_core/data.py:81`
- **Problem**: Adds 'DATE' column that could conflict with existing data
- **Current Code**:
  ```python
  df['DATE'] = pd.to_datetime(df[date_col])
  ```
- **Fix**:
  - [ ] Use a unique column name like `_PARSED_DATE`
  - [ ] Or filter without adding column: `df[pd.to_datetime(df[date_col]) >= ...]`
  - [ ] Or document this side effect clearly
- **Status**: ⬜ Not started

---

## 🟢 Improvements (Nice to Have)

### 9. Replace Print Statements with Logging
- **Files**: All modules
- **Problem**: `print()` can't be controlled, redirected, or suppressed
- **Tasks**:
  - [ ] Add `import logging` to each module
  - [ ] Create logger: `logger = logging.getLogger(__name__)`
  - [ ] Replace `print()` with `logger.info()`, `logger.debug()`, etc.
  - [ ] Add logging config to `config.py`
- **Status**: ⬜ Not started

---

### 10. Add CLI Arguments to Orchestrators
- **Files**: `analyses/*.py`
- **Problem**: Parameters hardcoded; requires code changes to customize
- **Tasks**:
  - [ ] Add `argparse` or `click` dependency
  - [ ] Add arguments for: `--data-file`, `--output-dir`, `--suppliers`, `--start-date`
  - [ ] Keep defaults from `config.py`
- **Example**:
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-file', default=config.DEFAULT_DATA_FILE)
  parser.add_argument('--output-dir', default=config.DEFAULT_OUTPUT_BASE)
  ```
- **Status**: ⬜ Not started

---

### 11. Use Dataclasses for Results
- **Files**: `efa_core/efa.py`, `efa_core/models.py`, `efa_core/stats.py`
- **Problem**: Functions return plain dicts; no type safety
- **Tasks**:
  - [ ] Create `efa_core/types.py` with result dataclasses
  - [ ] Define: `EFAResult`, `RFResult`, `ANOVAResult`, `FactorabilityResult`
  - [ ] Update functions to return dataclass instances
- **Example**:
  ```python
  @dataclass
  class EFAResult:
      factor_analyzer: FactorAnalyzer
      loadings: pd.DataFrame
      communalities: pd.DataFrame
      variance: pd.DataFrame
      n_factors: int
      rotation: str
  ```
- **Status**: ⬜ Not started

---

### 12. Add Unit Tests
- **Location**: Create `tests/` directory
- **Priority Tests**:
  - [ ] `tests/test_data.py`
    - `test_create_tier_labels_stable_ordering`
    - `test_filter_by_suppliers_handles_missing_column`
    - `test_standardize_features_handles_nan`
  - [ ] `tests/test_efa.py`
    - `test_run_efa_returns_correct_shape`
    - `test_check_factorability_with_singular_matrix`
  - [ ] `tests/test_models.py`
    - `test_train_rf_handles_nan_values`
    - `test_predict_with_model_matches_training`
- **Setup**:
  - [ ] Add `pytest` to requirements.txt
  - [ ] Create `tests/__init__.py`
  - [ ] Add test fixtures for sample data
- **Status**: ⬜ Not started

---

### 13. Centralize Visualization Parameters
- **Files**: `analyses/*.py`, `efa_core/viz.py`
- **Problem**: Magic numbers scattered in plot functions
- **Examples**:
  - `figsize=(10, 6)` - why this size?
  - `'█' * int(row['importance'] * 50)` - why 50?
  - `vmin=-1, vmax=1` - loading bounds
- **Tasks**:
  - [ ] Add `VIZ_CONFIG` dict to `config.py`
  - [ ] Include: `DEFAULT_FIGSIZE`, `HEATMAP_FIGSIZE`, `BAR_SCALE_FACTOR`
  - [ ] Update plot functions to use config values
- **Status**: ⬜ Not started

---

### 14. Fix Output Path Inconsistency
- **File**: `analyses/full_tier_analysis.py:377`
- **Problem**: Bypasses `output.save_report()` function
- **Current Code**:
  ```python
  with open(f"{output_dir}/{PIPELINE_NAME}-interpretation.txt", 'w') as f:
      f.write(interpretation)
  ```
- **Fix**:
  - [ ] Replace with: `output.save_report(interpretation, output_dir, PIPELINE_NAME, 'interpretation')`
- **Status**: ⬜ Not started

---

## 📋 Implementation Order

### Phase 1: Critical Fixes (Before Merge)
1. [ ] #1 - Stable tier labeling
2. [ ] #2 - Better NaN handling
3. [ ] #3 - Fix import pattern

### Phase 2: Stability (Next Sprint)
4. [ ] #6 - Input validation
5. [ ] #4 - Targeted warning suppression
6. [ ] #8 - Date column side effect

### Phase 3: Code Quality (Backlog)
7. [ ] #5 - Consistent return types
8. [ ] #7 - Avoid double scaling
9. [ ] #14 - Output path consistency

### Phase 4: Enhancements (Future)
10. [ ] #9 - Logging framework
11. [ ] #10 - CLI arguments
12. [ ] #11 - Dataclasses for results
13. [ ] #12 - Unit tests
14. [ ] #13 - Visualization config

---

## ✅ What's Already Good

- [x] Clear module separation (SRP)
- [x] Consistent file naming: `{DATE}-{test}-{suffix}.{ext}`
- [x] Comprehensive docstrings
- [x] Centralized config in `config.py`
- [x] Output directory auto-creation (`os.makedirs(exist_ok=True)`)
- [x] Preserved legacy code in `archive/` for reference

---

## Notes

- PR Link: https://github.com/cquehl/DataAnalytics/pull/2
- Branch: `feature/modular-framework-orchestrators`
- Review Date: 2026-02-09
