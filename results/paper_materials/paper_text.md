
# Filo-Priori V10: Experimental Results

## Abstract Snippet
We present Filo-Priori V10, a hybrid neuro-symbolic approach for test case
prioritization that combines neural feature learning with heuristic-based
scoring through residual learning. Our experiments on the RTPTorrent dataset
demonstrate that V10 achieves an APFD of 0.6856, representing a
15.7% improvement over the Recently-Failed baseline (APFD=0.5926).
The ablation study reveals that both neural and heuristic components contribute
significantly to the model's performance, with the learned fusion weight (α≈0.5)
suggesting an optimal balance between data-driven and expert-knowledge approaches.

## Results Section

### RQ1: How effective is V10 compared to baseline methods?

Table X presents the main results. V10-Full with LambdaRank loss achieves the
highest APFD of 0.6856, outperforming all baselines including
Recently-Failed (0.5926, +15.7%) and Failure-Rate
(0.5801, +18.2%). The Wilcoxon signed-rank test indicates
that V10-Full with APFD loss shows statistically significant improvement over
Recently-Failed (p=0.0312).

### RQ2: What is the contribution of each component?

The ablation study (Table Y) reveals that:
1. **Neural-only model** achieves APFD=0.6161, showing that neural
   feature learning alone provides a 4.0% improvement over
   the baseline.
2. **Heuristic-only model** achieves APFD=0.5221, performing
   -11.9% worse than the baseline, indicating that raw heuristic
   combination without learning is suboptimal.
3. **Full V10 model** achieves APFD=0.6751, demonstrating that the
   residual fusion provides an additional 9.6% improvement
   over the neural-only variant.

### RQ3: What is the optimal balance between neural and heuristic components?

The learned α value of approximately 0.50 suggests that the optimal
configuration balances neural and heuristic contributions equally. Fixed α
experiments show that:
- α=0.2 (neural-heavy): APFD=0.6399
- α=0.8 (heuristic-heavy): APFD=0.6251

Both perform worse than the learned α, validating the importance of adaptive
fusion weights.

### RQ4: Which loss function is more effective for ranking optimization?

Comparing LambdaRank and direct APFD loss:
- **LambdaRank**: APFD=0.6856
- **APFD Loss**: APFD=0.6751

LambdaRank achieves slightly better results (1.55% higher), suggesting
that pairwise ranking optimization provides marginal benefits over direct metric
optimization for this task.

## Key Findings

1. **Residual learning is effective**: Combining neural learning with heuristic
   priors through residual connections improves performance by 9.6%.

2. **Balanced fusion is optimal**: The model learns α≈0.5, indicating equal
   importance of neural and heuristic components.

3. **LambdaRank marginally outperforms APFD loss**: Pairwise ranking optimization
   provides a 1.55% improvement over direct metric optimization.

4. **Statistical significance achieved**: V10-Full-APFD shows statistically
   significant improvement over the Recently-Failed baseline (p=0.0312).
