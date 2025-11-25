# APFD Module: Consolidation and Fixes

**Date:** 2025-11-06  
**Affected Experiment:** experiment_017_ranking_corrected_03  
**Impact:** Code organization and Mean APFD calculation

---

## Summary

This document describes two major improvements to the APFD calculation system:

1. **Code Consolidation** - Merged duplicate APFD modules into single source of truth
2. **Business Rule Fix** - Corrected count_tc=1 → APFD=1.0 calculation

---

## Part 1: Code Consolidation

### Problem

The codebase had **two separate modules** for APFD calculation:
- `src/evaluation/apfd.py` (460 lines)
- `src/evaluation/apfd_calculator.py` (428 lines)

This caused:
- Confusion about which module to use
- Code duplication
- Circular import dependencies
- Maintenance difficulties

### Solution

**Consolidated all APFD functionality into single module:**
- `src/evaluation/apfd.py` (~510 lines)

### Changes Made

1. **Migrated** `count_total_commits()` from apfd_calculator.py to apfd.py
2. **Removed** import dependency on apfd_calculator
3. **Archived** redundant files:
   - `apfd_calculator.py` → `archive/evaluation/`
   - `test_refactoring.py` → `archive/`
4. **Moved** utility script:
   - `recalculate_apfd_fix_count_tc_1.py` → `scripts/`
5. **Created** documentation:
   - `src/evaluation/README.md`
   - `archive/evaluation/README.md`

### Benefits

- ✓ Single source of truth for APFD
- ✓ No circular dependencies
- ✓ Clearer code organization
- ✓ Easier maintenance
- ✓ Less confusion for developers

---

## Part 2: Business Rule Fix (count_tc=1)

### Problem

Builds with only 1 test case (count_tc=1) were receiving APFD=0.5 instead of APFD=1.0.

**Business Rule (from master_vini):**
```python
# Special case: single test case always returns 1.0
# With only one test, there's no ordering to optimize
if n == 1:
    return 1.0
```

**Rationale:** When there's only 1 test case, there's no ordering optimization possible. The test will be executed anyway, so APFD should be 1.0 (perfect).

### Impact

**Affected builds:** 23 out of 277 builds (8.3%)

**Before fix:**
```
Total builds: 277
Builds with count_tc=1: 23
Mean APFD: 0.555148
Builds with APFD=1.0: 0
```

**After fix:**
```
Total builds: 277
Builds with count_tc=1: 23
Mean APFD: 0.596664
Builds with APFD=1.0: 23
```

**Improvement:**
- Mean APFD: 0.555148 → 0.596664
- Absolute gain: +0.041516
- Percentage gain: **+7.48%**

### Affected Builds

All 23 builds now correctly have APFD=1.0:

```
T1TGN33.60-23    T1TZ33.3-60      T2SE33.73-8      T3TD33.10
U1SJ34.2-54      U1TD34.100-7     U1TT34.126       U1UD34.16-7
U2UAN34.72-35    U2UM34.27-8      U2UU34.17        U2UU34.40-5
U2UUI34.40-6     U3TZ34.2-58      U3UX34.30        U3UX34.9
UTPN34.141       UTR34.116        UTR34.173        UTT34.104
UUG34.20         UUU34.21         UUU34.25
```

### Implementation

The fix is implemented at two levels:

**1. Function level (calculate_apfd_single_build):**
```python
# Business rule: if only 1 test case, APFD = 1.0
if n_tests == 1:
    return 1.0
```

**2. Per-build level (calculate_apfd_per_build):**
```python
# CRITICAL BUSINESS RULE: count_tc=1 → APFD=1.0
# When there's only 1 test case, there's no ordering to optimize.
# The test will be executed anyway, so APFD should be 1.0 (perfect).
if count_tc == 1:
    # Still verify build has at least one failure
    [... verification code ...]
    
    # Add to results with APFD=1.0
    results.append({
        'apfd': 1.0,  # Business rule: count_tc=1 → APFD=1.0
        [...]
    })
    continue  # Skip standard APFD calculation
```

### Files Updated

**Fixed:**
- `results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv.csv`
  - Mean APFD updated to 0.596664

**Backup:**
- `results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv_OLD.csv`
  - Original file with Mean APFD 0.555148

**Script:**
- `scripts/recalculate_apfd_fix_count_tc_1.py`
  - Recalculation script with fix

---

## Current Module Structure

### src/evaluation/apfd.py

**Core Functions:**

1. `count_total_commits(df_build)` - Count unique commits including CRs
2. `calculate_apfd_single_build(ranks, labels)` - Calculate APFD for one build
3. `calculate_apfd_per_build(df, ...)` - Calculate APFD for all builds
4. `calculate_ranks_per_build(df, ...)` - Calculate priority ranks per build
5. `generate_apfd_report(df, ...)` - Generate complete APFD report
6. `generate_prioritized_csv(df, ...)` - Generate prioritized test cases CSV
7. `print_apfd_summary(summary_stats)` - Print formatted summary

**Business Rules:**

1. ✓ **count_tc=1 → APFD=1.0** - Single test case always gets APFD=1.0
2. ✓ **Only "Fail" results** - Only builds with ≥1 "Fail" result included
3. ✓ **Per-build calculation** - APFD calculated PER BUILD, not globally
4. ✓ **277 builds expected** - Validation of total build count

**Usage Example:**

```python
from src.evaluation.apfd import (
    calculate_ranks_per_build,
    generate_apfd_report,
    print_apfd_summary
)

# Calculate ranks per build
df = calculate_ranks_per_build(
    df, 
    probability_col='probability',
    build_col='Build_ID'
)

# Generate APFD report
results_df, summary_stats = generate_apfd_report(
    df,
    method_name="dual_stream_gnn_exp_17",
    test_scenario="full_test_csv_277_builds",
    output_path="results/experiment_017/apfd_per_build.csv"
)

# Print summary
print_apfd_summary(summary_stats)
```

---

## Verification

### Code Consolidation

```bash
# Test imports
python -c "
import sys
sys.path.insert(0, 'src/evaluation')
import apfd
print('✓ apfd.py imports successfully')
print('✓ count_total_commits:', hasattr(apfd, 'count_total_commits'))
print('✓ calculate_apfd_single_build:', hasattr(apfd, 'calculate_apfd_single_build'))
"
```

### count_tc=1 Fix

```bash
# Verify all builds with count_tc=1 have APFD=1.0
awk -F',' 'NR>1 && $4==1 {print $2,$4,$6}' \
  results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv.csv

# Output: All show APFD=1.0
T1TGN33.60-23 1 1.0
T1TZ33.3-60 1 1.0
[... 21 more builds ...]
UUU34.25 1 1.0
```

---

## Related Documentation

- `/docs/TC_COUNT_FIX.md` - Detailed count_tc=1 fix documentation
- `/src/evaluation/README.md` - Evaluation module guide
- `/archive/evaluation/README.md` - Archive documentation

---

## Migration Guide

### Before (Old Code)

```python
# DON'T USE - Old pattern
from src.evaluation.apfd_calculator import APFDCalculator

count = APFDCalculator.count_total_commits(df)
```

### After (New Code)

```python
# USE THIS - New consolidated pattern
from src.evaluation.apfd import count_total_commits

count = count_total_commits(df)
```

---

## Summary Statistics

**Code Consolidation:**
- Files reduced: 2 → 1
- Lines: 888 → 510
- Dependencies: Circular → None
- Clarity: Improved ✓

**Business Rule Fix:**
- Builds affected: 23/277 (8.3%)
- Mean APFD improvement: +7.48%
- Correct APFD=1.0: 0 → 23 builds
- Alignment with master_vini: ✓

---

**Last Updated:** 2025-11-06  
**Status:** Complete ✓  
**Experiment:** experiment_017_ranking_corrected_03
