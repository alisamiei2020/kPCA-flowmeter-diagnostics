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

- **Constant variables.** The 11 variables with std = 0 in the healthy class are removed from the kPCA
  analysis, because a zero-variance variable carries no direction of variation. They are meter
  performance, signal quality ×8, and gain P1A/P1B. All 11 *do* vary under both faults, so they are
  analysed separately later. **40 variables remain.**
- **Autoscaling.** Each variable is z-scored with the mean and sample std (n−1 denominator) over all
  92 samples. No epsilon: a zero std raises an error.
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
| **M1** | `01_data_exploration.ipynb` | names, `load_meter_b`, `constant_columns`, `zscore` | assert cell passes; class-balance, class-mean and correlation figures |
| **M2** | `02_pca_baseline.ipynb` | `biplot` (scores + loading arrows) | our PCA matches `sklearn.decomposition.PCA` up to sign; scree plot; biplots PC1–2, 1–3, 2–3 |
| **M3** | `03_kpca.ipynb` | `kernel_matrix`, `kpca` wrapper around `KernelPCA` | linear kPCA = PCA scores; centred kernel matrix has rows summing to 0; score plots for each kernel |
| **M4** | `04_kpca_biplots.ipynb` | `correlation_loadings` (corr. of each variable with each kPC) | for the linear kernel, the arrows match the M2 PCA loadings; multiple biplots for each kernel |
| **M5** | `05_kernel_comparison.ipynb` | separation scores: silhouette, Fisher ratio, CV k-NN accuracy | comparison table and figure over kernel settings; fixed random seed |
| **M6** | `report/report.md` | – | every notebook runs top to bottom; the report answers both questions |
| *Extra* | – | variables set aside in M1; PyTorch kernel learning; Matérn; T² charts | if time allows |

## Working rules
One milestone at a time; plan → approval → build → run → show results → proposed commit message.
