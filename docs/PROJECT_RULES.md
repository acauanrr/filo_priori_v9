# Filo-Priori Project Rules and Guidelines

**Established:** 2025-11-10
**Version:** 1.0
**Status:** ACTIVE

---

## Core Principles

### 1. **Single Codebase, Incremental Development**

❌ **DO NOT:**
- Create versioned copies (v8, v9, v10, etc.)
- Duplicate `main_vX.py` files
- Create `src_v2/` directories
- Fork the entire project for changes

✅ **DO:**
- Make changes directly in the main codebase
- Use git branches for experimental features
- Keep a single `main.py` file
- Evolve code incrementally

**Rationale:** Multiple versions pollute the repository, create confusion, and make maintenance impossible.

---

### 2. **Experiment Numbering System**

All experiments are numbered sequentially and automatically:

```
results/
├── experiment_001/    # First experiment
├── experiment_002/    # Second experiment
├── experiment_003/    # Third experiment
└── ...
```

**Automatic Numbering:**
- `./run_experiment.sh` automatically detects the next available number
- No manual numbering required
- No gaps in numbering sequence

**Experiment Directory Structure:**
```
experiment_XXX/
├── config_used.yaml           # Config snapshot
├── command.txt                # Exact command run
├── timestamps.txt             # Start/end times
├── output.log                 # Full execution log
├── apfd_per_build.csv        # APFD results
├── prioritized_test_cases.csv
├── confusion_matrix.png
└── ...
```

---

### 3. **Configuration Management**

**Active Config:**
- `configs/experiment.yaml` - The ONLY active configuration file

**Config Updates:**
- Edit `configs/experiment.yaml` directly for new experiments
- Old configs are automatically archived in experiment directories

**Archived Configs:**
- `archive/configs/` - Historical configurations for reference
- Never modify archived configs

**DO NOT:**
- Create `experiment_v9_qodo.yaml`, `experiment_final.yaml`, etc.
- Keep multiple "active" configs
- Version config files

---

### 4. **File Organization**

#### **Active Files** (Project Root)
```
filo_priori_v8/
├── main.py                    # Single unified entry point
├── setup_experiment.sh        # Environment setup
├── run_experiment.sh          # Experiment runner
├── requirements.txt           # Core dependencies
├── requirements_v9.txt        # Additional dependencies
├── README.md                  # Main documentation
├── PROJECT_RULES.md          # This file
└── MIGRATION_V8_TO_V9.md     # Migration guide
```

#### **Source Code**
```
src/
├── embeddings/
│   ├── qodo_encoder.py       # Active encoder
│   └── semantic_encoder.py   # Legacy (kept for reference)
├── preprocessing/
│   ├── commit_extractor.py   # Active
│   └── ...
└── ...
```

#### **Archive** (Obsolete Files)
```
archive/
├── docs/                     # Old documentation
├── scripts/                  # Old scripts
├── configs/                  # Old configurations
├── old_mains/               # Old main_vX.py files
└── test_files/              # Old test scripts
```

**Archive Rules:**
- Files in `archive/` are READ-ONLY
- Never import from `archive/`
- Keep for historical reference only
- Can be deleted if confirmed obsolete

---

### 5. **Code Changes**

#### **Making Changes:**

1. **Small Changes:** Edit files directly
   ```bash
   vim src/models/dual_stream_v8.py
   git add src/models/dual_stream_v8.py
   git commit -m "Fix: Update input dimension to 3072"
   ```

2. **Large Changes:** Use feature branches
   ```bash
   git checkout -b feature/attention-fusion
   # Make changes
   git commit -m "Add attention-based fusion layer"
   git checkout main
   git merge feature/attention-fusion
   ```

3. **Breaking Changes:** Document in commit
   ```bash
   git commit -m "BREAKING: Replace BGE with Qodo-Embed

   - Removes BGE encoder
   - Adds QodoEncoder
   - Updates config schema
   - See MIGRATION_V8_TO_V9.md"
   ```

#### **DO NOT:**
- Copy `main.py` to `main_new.py`
- Create `src2/` or `src_backup/`
- Comment out large code blocks (delete instead)
- Keep `old_function_v2()` alongside `new_function()`

---

### 6. **Dependencies**

**All Dependencies:** `requirements.txt`
- PyTorch, PyTorch Geometric
- Transformers
- sentence-transformers (for Qodo-Embed)
- NumPy, Pandas, etc.

**Installing:**
```bash
./setup_experiment.sh   # Installs all dependencies
```

**Adding New Dependencies:**
1. Add to appropriate requirements file
2. Test installation
3. Update `setup_experiment.sh` if needed
4. Commit changes

---

### 7. **Running Experiments**

#### **Standard Workflow:**

```bash
# 1. Setup (run once)
./setup_experiment.sh

# 2. Edit config
vim configs/experiment.yaml

# 3. Run experiment (auto-numbered)
./run_experiment.sh

# 4. Review results
cat results/experiment_XXX/output.log
```

#### **Custom Runs:**

```bash
# Use custom config
./run_experiment.sh --config configs/custom.yaml

# Force CPU
./run_experiment.sh --device cpu

# Test on sample
./run_experiment.sh --sample 1000
```

#### **DO NOT:**
- Run `python main.py` directly (use `run_experiment.sh`)
- Manually create `results/experiment_XXX/`
- Edit files in `results/`
- Delete experiment directories

---

### 8. **Documentation**

#### **Required Documentation:**

- `README.md` - Main project documentation
- `PROJECT_RULES.md` - This file
- Code comments for complex logic
- Docstrings for all functions/classes

#### **Optional Documentation:**

- Migration guides (e.g., `MIGRATION_V8_TO_V9.md`)
- Architecture diagrams
- Experiment reports

#### **Archive Old Docs:**

When documentation becomes obsolete:
```bash
mv OLD_DOC.md archive/docs/
```

---

### 9. **Code Quality**

#### **Standards:**

- **Python:** PEP 8 style guide
- **Line length:** Max 100 characters
- **Imports:** Organized (stdlib, third-party, local)
- **Type hints:** Use for function signatures
- **Docstrings:** Google style

#### **Example:**

```python
def encode_texts(
    self,
    texts: List[str],
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode a list of texts to embeddings.

    Args:
        texts: List of text strings to encode
        show_progress: Whether to show progress bar

    Returns:
        Numpy array of embeddings [num_texts, embedding_dim]
    """
    ...
```

---

### 10. **Git Workflow**

#### **Commit Messages:**

Format:
```
<type>: <subject>

<body>
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code refactoring
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance

Examples:
```bash
git commit -m "feat: Add Qodo-Embed encoder"
git commit -m "fix: Correct APFD calculation for edge cases"
git commit -m "refactor: Simplify commit extraction logic"
```

#### **Branching:**

- `main` - Stable, working code
- `feature/X` - New features
- `fix/X` - Bug fixes
- `experiment/X` - Experimental code

**Merge only working code to main.**

---

### 11. **Cleanup Policy**

#### **When to Archive:**

Move to `archive/` if:
- File is obsolete (replaced by newer version)
- Script is no longer used
- Documentation is outdated

#### **When to Delete:**

Delete if:
- File is temporary (cache, logs)
- Generated output (can be reproduced)
- Confirmed useless after 30 days in archive

#### **Monthly Cleanup:**

Review `archive/` and delete confirmed obsolete files.

---

### 12. **Cache and Logs**

#### **Cache:**
```
cache/
├── embeddings_qodo/
│   ├── train_tc_embeddings.npy
│   └── ...
└── structural_features.pkl
```

**Rules:**
- Cache is for speed, not persistence
- Can be deleted anytime (will regenerate)
- Add to `.gitignore`

#### **Logs:**
```
logs/
└── experiment_XXX.log   # Historical logs
```

**Rules:**
- Logs in `results/experiment_XXX/` are permanent
- Logs in `logs/` can be deleted after 30 days
- Don't commit logs to git

---

### 13. **Testing**

#### **Before Committing:**

1. Run basic test:
   ```bash
   ./run_experiment.sh --sample 100
   ```

2. Check output:
   ```bash
   tail -100 results/experiment_XXX/output.log
   ```

3. Verify no errors

#### **Integration Tests:**

Run full experiment on small dataset before committing major changes.

---

### 14. **Collaboration**

#### **Pull Requests:**

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit PR with description
5. Review and merge

#### **Code Review:**

- Review all PRs before merging
- Check for rule violations
- Verify tests pass
- Ensure documentation updated

---

### 15. **Violations**

If you find code violating these rules:

1. Fix immediately if minor
2. Create issue for major violations
3. Update this document if rules are unclear

**Common Violations:**

- Creating `main_v10.py` → Use `main.py`
- Manual experiment numbers → Use `run_experiment.sh`
- Committing cache files → Add to `.gitignore`
- Undocumented functions → Add docstrings

---

## Summary Checklist

Before committing code:

- [ ] No versioned files (v8, v9, etc.)
- [ ] Single `main.py` file
- [ ] Configs in `configs/experiment.yaml`
- [ ] Old files moved to `archive/`
- [ ] Code documented
- [ ] Tests pass
- [ ] Git commit message follows format

---

## Questions?

If these rules are unclear or need updating:

1. Create an issue
2. Propose changes
3. Update this document
4. Commit updated rules

---

**Remember:** Clean code is happy code! 🎉

**Last Updated:** 2025-11-10
**Next Review:** 2025-12-10
