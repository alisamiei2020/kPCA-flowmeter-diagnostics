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

| Kernel | Formula | Source in scikit-learn |
|---|---|---|
| Linear | k = xᵀy | `KernelPCA(kernel="linear")`: the same as ordinary PCA, used as the reference |
| Polynomial | k = (γ xᵀy + c)^d | `KernelPCA(kernel="poly", degree=d, gamma=γ, coef0=c)` |
| Gaussian (RBF) | k = exp(−‖x−y‖² / 2σ²) | `KernelPCA(kernel="rbf", gamma=1/(2σ²))` |
| Cauchy | k = 1 / (1 + ‖x−y‖²/σ²) | `RationalQuadratic(length_scale=ℓ, alpha=1)` with σ² = 2ℓ², passed as `kernel="precomputed"` |
| *(extra)* Matérn | see paper | `sklearn.gaussian_process.kernels.Matern`, also precomputed |

`KernelPCA` centres the kernel matrix and does the eigendecomposition itself. In the notebook we explain
the centring formula K_c = (I − 1/n)K(I − 1/n), and check once that our understanding is right: the
linear-kernel scores must equal PCA scores. The width σ is started at the median pairwise distance of
the autoscaled data (the "median heuristic"), then varied.

## Milestones

| # | Notebook | Adds to `kpca_tools.py` | Verified by |
|---|---|---|---|
| **M1** | `01_data_exploration.ipynb` | names, `load_meter_b`, `constant_columns` (max = min), `zscore` | assert cell passes; class-balance, class-mean and correlation figures |
| **M2** | `02_pca_baseline.ipynb` | `biplot` (scores + loading arrows) | our PCA matches `sklearn.decomposition.PCA` up to sign; scree plot; biplots PC1–2, 1–3, 2–3 |
| **M3** | `03_kpca.ipynb` | `kernel_matrix`, `kpca` wrapper around `KernelPCA` | linear kPCA = PCA scores; centred kernel matrix has rows summing to 0; score plots for each kernel |
| **M4** | `04_kpca_biplots.ipynb` | `correlation_loadings` (corr. of each variable with each kPC) | for the linear kernel, the arrows match the M2 PCA loadings; multiple biplots for each kernel |
| **M5** | `05_kernel_comparison.ipynb` | separation scores: silhouette, Fisher ratio, CV k-NN accuracy | comparison table and figure over kernel settings; fixed random seed |
| **M6** | `report/report.md` | – | every notebook runs top to bottom; the report answers both questions |
| *Extra* | – | healthy-constant variables on their own; PyTorch kernel learning; Matérn; T² charts | if time allows |

## Working rules
One milestone at a time; plan → approval → build → run → show results → proposed commit message.
