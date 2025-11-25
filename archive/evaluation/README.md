# Archive: APFD Calculation Modules

## Consolidation (2025-11-06)

This directory contains archived APFD calculation modules that have been consolidated into a single module for better code maintainability.

### Files Archived

1. **apfd_calculator.py** - Enhanced APFD calculator with multiple calculation methods
   - Contained: APFDCalculator class with various APFD calculation methods
   - Methods: calculate_classic_apfd(), calculate_weighted_apfd(), calculate_napfd(), etc.
   - Reason for archival: Only `count_total_commits()` was used in production code

### Consolidation Summary

**Before:**
- `src/evaluation/apfd.py` - Main APFD module (460 lines)
- `src/evaluation/apfd_calculator.py` - APFDCalculator class (428 lines)
- Total: 888 lines across 2 files
- Problem: Code duplication and confusion about which module to use

**After:**
- `src/evaluation/apfd.py` - Consolidated APFD module (~510 lines)
- Total: 1 file with all functionality
- Benefits:
  - Single source of truth for APFD calculations
  - No import dependencies between APFD modules
  - Clearer code organization
  - Easier maintenance

### What Was Migrated

From `apfd_calculator.py` to `apfd.py`:
- `count_total_commits()` function - Count unique commits (including CRs)

### What Was NOT Migrated (and Why)

The following methods from APFDCalculator were NOT migrated because they were not used in production:
- `calculate_classic_apfd()` - Classic APFD formula
- `calculate_weighted_apfd()` - Weighted APFD for different failure types
- `calculate_napfd()` - Normalized APFD with test costs
- `calculate_apfd_with_severity()` - Severity-weighted APFD
- `calculate_apfd_c()` - APFD with confidence scores
- `calculate_modified_apfd()` - Modified APFD with multiple variants
- `filter_builds_with_failures()` - Filter builds with failures

These methods remain available in this archived file if needed in the future.

### Current Production Module

**src/evaluation/apfd.py** now contains:

**Core Functions:**
- `count_total_commits()` - Count unique commits (migrated from apfd_calculator.py)
- `calculate_apfd_single_build()` - Calculate APFD for a single build
- `calculate_apfd_per_build()` - Calculate APFD for all builds
- `calculate_ranks_per_build()` - Calculate priority ranks per build
- `generate_apfd_report()` - Generate complete APFD report
- `generate_prioritized_csv()` - Generate prioritized test cases CSV
- `print_apfd_summary()` - Print formatted APFD summary

**Critical Business Rules Implemented:**
1. count_tc=1 → APFD=1.0 (when only 1 test case, APFD is always 1.0)
2. Only builds with at least 1 "Fail" result are included
3. APFD is calculated PER BUILD, not globally
4. Expects exactly 277 builds (project requirement)

### Usage

If you need to use the archived APFDCalculator methods:

```python
import sys
sys.path.insert(0, 'archive/evaluation')
from apfd_calculator import APFDCalculator

# Use methods
apfd = APFDCalculator.calculate_classic_apfd(df_ordered)
```

However, for production use, always use:

```python
from src.evaluation.apfd import (
    calculate_apfd_per_build,
    calculate_apfd_single_build,
    count_total_commits,
    generate_apfd_report
)
```

## Related Documentation

- `/docs/TC_COUNT_FIX.md` - Documentation of count_tc=1 → APFD=1.0 fix
- `/src/evaluation/apfd.py` - Current production APFD module

---
**Date:** 2025-11-06
**Reason:** Code consolidation to eliminate duplication and ambiguity
