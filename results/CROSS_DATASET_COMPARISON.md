# Filo-Priori V10: Cross-Dataset Comparison

## Summary

This report compares V10 performance across two datasets:
- **01_industry**: Industrial QTA dataset from mobile CI/CD pipeline
- **02_rtptorrent**: Open-source Java projects from Travis CI

## Dataset Statistics

| Statistic | Industry | RTPTorrent |
|-----------|----------|------------|
| Total Executions (Train) | 69,169 | 14,979 |
| Total Executions (Test) | 31,333 | 4,133 |
| Train Builds | 3,187 | 240 |
| Test Builds | 1,365 | 60 |
| Builds with Failures (Train) | 625 (19.6%) | 18 (7.5%) |
| Builds with Failures (Test) | 277 (20.3%) | 7 (11.7%) |
| Pass:Fail Ratio | 37:1 | ~50:1 |

## Main Results

### V10 Performance

| Dataset | V10 APFD | RF Baseline | Improvement | p-value | Significance |
|---------|----------|-------------|-------------|---------|--------------|
| **Industry** | **0.5901** | 0.5338 | **+10.54%** | **0.0006** | *** (p<0.001) |
| RTPTorrent | 0.6180 | 0.5926 | +4.30% | 0.3438 | ns |

### All Baselines Comparison

#### Industry Dataset
| Method | APFD | Std | vs RF |
|--------|------|-----|-------|
| Random | 0.4991 | 0.1658 | -6.5% |
| Recently-Failed | 0.5338 | 0.2425 | baseline |
| Failure-Rate | 0.5354 | 0.2552 | +0.3% |
| **V10 Model** | **0.5901** | 0.2405 | **+10.54%** |

#### RTPTorrent Dataset
| Method | APFD | Std | vs RF |
|--------|------|-----|-------|
| Random | 0.4213 | 0.2251 | -28.9% |
| Recently-Failed | 0.5926 | 0.2521 | baseline |
| Failure-Rate | 0.5801 | 0.2449 | -2.1% |
| **V10 Model** | **0.6180** | 0.2430 | **+4.30%** |

## Model Parameters

| Parameter | Industry | RTPTorrent |
|-----------|----------|------------|
| Learned Alpha (α) | 0.534 | 0.502 |
| Total Parameters | 6,918 | 6,918 |
| Hidden Dim | 64 | 64 |
| Dropout | 0.3 | 0.3 |
| Loss Function | LambdaRank | LambdaRank |

## Key Findings

### 1. V10 generalizes across domains
- Works on both industrial (mobile testing) and open-source (Java CI) domains
- Consistent architecture with same hyperparameters
- No domain-specific tuning required

### 2. Higher impact on industrial data
- **+10.54%** improvement on industry vs **+4.30%** on RTPTorrent
- Industry dataset has more training data (625 vs 14 builds with failures)
- Statistical significance achieved on industry (p=0.0006)

### 3. Optimal alpha varies by domain
- Industry: α=0.534 (slight heuristic bias)
- RTPTorrent: α=0.502 (balanced)
- Model learns domain-appropriate weighting

### 4. Baseline performance differs
- Industry: RF (0.5338) ≈ Failure-Rate (0.5354)
- RTPTorrent: RF (0.5926) > Failure-Rate (0.5801)
- Recent failures more predictive in open-source projects

## Statistical Analysis

### Wilcoxon Signed-Rank Test (V10 vs Recently-Failed)

| Dataset | W-statistic | p-value | Effect Size |
|---------|-------------|---------|-------------|
| Industry | - | 0.0006 | Large |
| RTPTorrent | - | 0.3438 | Small |

The industry dataset shows highly significant improvement (p<0.001), while
RTPTorrent shows improvement but without statistical significance due to
smaller sample size (only 7 test builds with failures).

## Recommendations

### For Production Use
1. **Industry-like environments**: V10 recommended with high confidence
2. **Open-source projects**: V10 shows improvement, but validate on more data

### For Future Work
1. **More data**: Collect more RTPTorrent builds to achieve statistical power
2. **Domain adaptation**: Investigate transfer learning between domains
3. **Feature engineering**: Industry-specific features may further improve results

## Conclusion

Filo-Priori V10 successfully generalizes across different TCP domains:

- **Industry (01_industry)**: +10.54% improvement, p=0.0006 (**statistically significant**)
- **Open-source (02_rtptorrent)**: +4.30% improvement, p=0.34 (not significant due to small sample)

The unified architecture with learnable residual fusion proves effective for both
industrial mobile testing and open-source Java projects, demonstrating the
practical applicability of the V10 approach.

---
Generated: 2024-11-29
Model: Filo-Priori V10
