# Project guideline – kPCA fault discrimination on Meter B

**Question.** Can kernel PCA separate the three operating states of a liquid ultrasonic flow meter
(healthy, gas injection, waxing), and which kernel separates them best?

**Deliverables.** Jupyter notebooks (one per milestone, saved with outputs) plus a 3–5 page report
in Markdown.

## Layout

```
data/raw/Meter_B.txt        raw data: 92 rows, 51 features + class
main/kpca_tools.py          shared functions, imported by every notebook (grows per milestone)
main/0X_*.ipynb             one notebook per milestone
main/figures/               PNGs, reused in the report
docs/GUIDELINE.md           this plan
report/report.md            final report (M6)
references/                 course papers (Duma et al. 2026)
```

The layout follows the reference toolbox `sab-in-science/k-mspc`: one shared code file plus notebooks
that call it.

## Preprocessing decisions (from M1)

- **Autoscaling over the whole data set.** Each variable is z-scored with the mean and sample std
  (n−1 denominator) over all 92 samples. No epsilon: a zero std raises an error. Reason: we ask whether
  kPCA on all the data separates the three states, so each variable gets equal weight over the full
  range it covers. Healthy-only scaling belongs to process monitoring, a different task; here it would
  also inflate a few variables (max |z| ≈ 8,000) until they dominate every kernel distance.
- **Constant variables.** Consistent with whole-data scaling, only variables constant over the whole
  data set are removed (tested as max = min, not std = 0, which floating-point rounding can miss).
  None are constant, so **all 51 variables are kept**.
- **Constant only in the healthy class:** 17 variables (meter performance, signal quality ×8, gain ×8).
  They are kept; they change only under faults, so they are likely strong fault indicators.
- **Kept on purpose:** the 15 waxing samples with a full path-4 dropout, and the duplicated gain A/B
  columns. Both are noted for the discussion.

## Kernels – library, not from scratch

All kernels and kPCA come from scikit-learn; we only write a thin wrapper and the plotting.

| Kernel | Formula | Source in scikit-learn (via `kpca_tools.kernel_matrix`) |
|---|---|---|
| Linear | k = xᵀy | `linear_kernel`: the same as ordinary PCA, used as an exact check |
| Polynomial | k = (γ xᵀy + c)^d | `polynomial_kernel` (default γ = 1/p, c = 1) |
| Gaussian (RBF) | k = exp(−‖x−y‖² / 2σ²) | `rbf_kernel` with γ = 1/(2σ²) |
| Cauchy | k = 1 / (1 + ‖x−y‖²/σ²) | `RationalQuadratic(length_scale=σ/√2, alpha=1)` |
| *(extra)* Matérn | see paper | `sklearn.gaussian_process.kernels.Matern` |

kPCA itself is `KernelPCA(kernel="precomputed")`, which centres the kernel matrix
(K_c = (I − 1/n)K(I − 1/n)) and does the eigendecomposition. Default width: σ = median distance between
samples.

## Choosing the kernel settings (M4)

- **Criterion: balanced kNN accuracy only** (k = 5) on the first 2 kPCs (3 and 7 reported too), with
  repeated stratified 5-fold cross-validation (10 repeats → 50 train/test splits; each sample
  tested 10 times) and a fixed random seed. Visual separation is judged in
  the biplots, not by a metric.
- **Everything is refitted on each training fold:** z-scoring (same `zscore`, n − 1), the median distance
  that sets σ, and kPCA. Test samples are only transformed.
- **Grid:** σ = median distance × 2^k, k = −4 … 4 (17 values); polynomial degree d = 1 … 5.
- **Tie rule:** among settings within one standard error of the best accuracy, take the smoothest
  (largest σ, lowest degree).
- **Limitation:** choosing and scoring on the same cross-validation is slightly optimistic; nested CV
  is an optional extra.
  The standard error (std of the 50 split scores / √50) treats the splits as independent, but the
  10 repeats reuse the same samples, so it understates the uncertainty somewhat.

## Milestones

| # | Notebook | Adds to `kpca_tools.py` | Verified by |
|---|---|---|---|
| **M1** | `01_data_exploration.ipynb` | names, `load_meter_b`, `constant_columns` (max = min), `zscore` | assert cell passes; class-balance, class-mean and correlation figures |
| **M2** | `02_pca_baseline.ipynb` | `pca`, `biplot` (loading or correlation arrows) | check cell (T = ZP, var(T) = λ, corr = p√λ); scree; score matrix PC1–7; loading and correlation biplots for PC1–2 … PC6–7 |
| **M3** | `03_kpca.ipynb` | `median_distance`, `kernel_matrix`, `kpca` | linear kPCA = PCA; centring and eigenvalues match formulas; kernel matrices, scree, score plots per kernel (default settings) |
| **M4** | `04_kernel_settings.ipynb` | `cv_balanced_accuracy` (everything refitted per fold) | accuracy curves over σ / degree per kernel; chosen setting per kernel; fixed random seed |
| **M5** | `05_kpca_biplots.ipynb` | `correlation_loadings`, `score_matrix`, `biplot_grid` | linear-kernel arrows match M2; kPC1–2 comparison of 4 kernels; score matrix and biplots kPC1–2 … kPC6–7 for poly d=2, Gaussian and Cauchy at default settings (option B); confusion matrices; final comparison |
| **M6** | `report/report.md` | – | every notebook runs top to bottom; the report answers both questions |
| *Extra* | – | healthy-constant variables on their own; PyTorch kernel learning; Matérn; T² charts | if time allows |

## Working rules
One milestone at a time; plan → approval → build → run → show results → proposed commit message.
