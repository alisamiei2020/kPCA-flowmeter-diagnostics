"""Shared functions for the Meter B kPCA project.

Each notebook imports this module; it grows one milestone at a time.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA, KernelPCA
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KernelCenterer

# ---------------------------------------------------------------------------
# Data description (UCI "Ultrasonic flowmeter diagnostics", Meter B)
# ---------------------------------------------------------------------------
PATHS = [1, 2, 3, 4]
ENDS = ["A", "B"]  # "both ends" columns are ordered P1A, P1B, P2A, ... (see notebook 01)


def _per_path(long, short):
    return [f"{long} P{p}" for p in PATHS], [f"{short}{p}" for p in PATHS]


def _per_end(long, short):
    return ([f"{long} P{p}{e}" for p in PATHS for e in ENDS],
            [f"{short}{p}{e}" for p in PATHS for e in ENDS])


def _build_names():
    long, short = [], []

    def add(l, s):
        long.extend(l)
        short.extend(s)

    add(["Profile factor", "Symmetry", "Crossflow", "Swirl angle"],
        ["PF", "SYM", "CF", "SWA"])                                   # 1-4
    add(*_per_path("Flow velocity", "V"))                           # 5-8
    add(["Average flow velocity"], ["Vavg"])                        # 9
    add(*_per_path("Speed of sound", "SoS"))                        # 10-13
    add(["Average speed of sound"], ["SoSavg"])                     # 14
    add(*_per_end("Signal strength", "SS"))                         # 15-22
    add(*_per_path("Turbulence", "TU"))                             # 23-26
    add(["Meter performance"], ["PERF"])                            # 27
    add(*_per_end("Signal quality", "SQ"))                          # 28-35
    add(*_per_end("Gain", "G"))                                     # 36-43
    add(*_per_end("Transit time", "TT"))                            # 44-51
    return long, short


FEATURE_NAMES, SHORT_NAMES = _build_names()

CLASS_NAMES = {1: "Healthy", 2: "Gas injection", 3: "Waxing"}

# Fixed class colours, used in every figure (first three categorical slots of a
# colour-blind-checked palette: blue, orange, aqua).
CLASS_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

# Diverging map for signed quantities (z-scores, correlations): blue - grey - red.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])

ARROW_COLOR = "#52514e"   # neutral grey: arrows are not a class
SURFACE = "#fcfcfb"       # thin ring around points so overlapping points stay visible

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Meter_B.txt"


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------
def load_meter_b(path=DATA_PATH):
    """Return X (n x 51 features) and y (n,) integer class labels."""
    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1].astype(int)


def constant_columns(X):
    """Indices of columns that take a single value (max == min).

    Tested with max - min instead of std == 0: a floating-point std of a
    constant column can come out as ~1e-15 instead of exactly 0.
    """
    return np.where(np.ptp(X, axis=0) == 0)[0]


def zscore(X):
    """Autoscale columns to mean 0 and sample std 1 (ddof=1).

    No epsilon is added: constant columns must be removed before calling this,
    so a zero std raises an error instead of being silently hidden.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    if np.any(sigma == 0):
        raise ValueError(f"constant columns {np.where(sigma == 0)[0]}; remove them first")
    return (X - mu) / sigma, mu, sigma


# ---------------------------------------------------------------------------
# PCA and biplots
# ---------------------------------------------------------------------------
def pca(Z):
    """Linear PCA of autoscaled data Z (n x p) with scikit-learn.

    Returns scores T (n x k), loadings P (p x k, unit-length columns),
    eigenvalues lam (k,) = variance of each score column, and the
    explained-variance ratio (k,).
    """
    model = PCA().fit(Z)
    return (model.transform(Z), model.components_.T,
            model.explained_variance_, model.explained_variance_ratio_)


def biplot(ax, T, A, y, names, pcs=(0, 1), n_arrows=15, circle=False, evr=None, prefix="PC"):
    """Scores T coloured by class, plus arrows A (one row per variable).

    A is either the loadings P or the correlations P * sqrt(lam). Only the
    `n_arrows` longest arrows in the chosen plane are drawn. Arrows are
    multiplied by one factor so they fit the score cloud; with circle=True
    that factor maps correlation 1 to the drawn circle. Returns the factor.
    """
    i, j = pcs
    for c in CLASS_NAMES:
        m = y == c
        ax.scatter(T[m, i], T[m, j], s=22, color=CLASS_COLORS[c], label=CLASS_NAMES[c],
                   edgecolor=SURFACE, linewidth=0.5, zorder=2)

    length = np.hypot(A[:, i], A[:, j])
    top = np.argsort(-length)[:n_arrows]
    reach = 0.9 * np.abs(T[:, [i, j]]).max()
    scale = reach if circle else reach / length[top].max()

    if circle:
        ax.add_patch(Circle((0, 0), scale, fill=False, color=ARROW_COLOR,
                            linewidth=0.8, linestyle="--", zorder=1))

    # identical variables (e.g. duplicated gain A/B columns) share one arrow and one joined label
    arrows = {}
    for k in top:
        arrows.setdefault((round(A[k, i], 12), round(A[k, j], 12)), []).append(names[k])

    tips = []
    for (ai, aj), labels in arrows.items():
        x, z = scale * ai, scale * aj
        ax.annotate("", xy=(x, z), xytext=(0, 0), zorder=3, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1, mutation_scale=8))
        ax.text(1.08 * x, 1.08 * z, "/".join(labels), fontsize=7, ha="center", va="center", zorder=4)
        tips.append((1.15 * x, 1.15 * z))
    ax.update_datalim(tips)   # arrows do not enlarge the axes by themselves
    ax.autoscale_view()

    ax.axhline(0, color="#d6d5d0", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#d6d5d0", linewidth=0.6, zorder=0)
    lab = [f"{prefix}{p + 1}" + (f" ({evr[p]:.1%})" if evr is not None else "") for p in pcs]
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    return scale



def biplot_grid(T, A, y, names, evr, title, planes=None, circle=False, prefix="PC", n_arrows=15):
    """Biplots for several planes (default PC1-2, PC2-3, ..., PC6-7) in a 2 x 3 grid."""
    import matplotlib.pyplot as plt
    planes = planes or [(k, k + 1) for k in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, pcs in zip(axes.flat, planes):
        biplot(ax, T, A, y, names, pcs=pcs, n_arrows=n_arrows, circle=circle, evr=evr, prefix=prefix)
        ax.set_aspect("equal", adjustable="datalim")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def score_matrix(T, y, evr, title, n_pc=7, prefix="PC"):
    """Scatter plots of the scores for every pair of the first n_pc components."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_pc, n_pc, figsize=(13, 13))
    for a in range(n_pc):
        for b in range(n_pc):
            ax = axes[a, b]
            if a == b:
                ax.text(0.5, 0.5, f"{prefix}{a + 1}\n{100 * evr[a]:.1f}%", ha="center", va="center",
                        fontsize=11, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[:].set_visible(False)
            elif a > b:
                for c in CLASS_NAMES:
                    m = y == c
                    ax.scatter(T[m, b], T[m, a], s=9, color=CLASS_COLORS[c],
                               edgecolor=SURFACE, linewidth=0.3, label=CLASS_NAMES[c])
                ax.tick_params(labelsize=6)
            else:
                ax.axis("off")
            if a == n_pc - 1 and b < n_pc - 1:
                ax.set_xlabel(f"{prefix}{b + 1}")
            if b == 0 and a > 0:
                ax.set_ylabel(f"{prefix}{a + 1}")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.9, 0.85), frameon=False, fontsize=11,
               markerscale=2)
    fig.suptitle(title, y=0.92, fontsize=13)
    return fig


def correlation_loadings(Z, T):
    """Pearson correlation of every variable (columns of Z) with every score column of T (p x k).

    Used as biplot arrows for kPCA, which has no loadings. For linear PCA on
    autoscaled data this equals P * sqrt(lambda).
    """
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    Ts = (T - T.mean(axis=0)) / T.std(axis=0, ddof=1)
    return Zs.T @ Ts / (len(Z) - 1)

# ---------------------------------------------------------------------------
# Kernel PCA
# ---------------------------------------------------------------------------
def median_distance(Z):
    """Median Euclidean distance over all pairs of samples (rows of Z)."""
    return np.median(pdist(Z))


def kernel_matrix(Z, kind, Y=None, sigma=None, degree=None, gamma=None, coef0=1.0):
    """Kernel matrix k(z_i, y_j) between the rows of Z and Y (Y = Z if omitted).

    kind: "linear"     x'y
          "poly"       (gamma x'y + coef0)^degree      (gamma defaults to 1/p)
          "rbf"        exp(-||x - y||^2 / (2 sigma^2))
          "cauchy"     1 / (1 + ||x - y||^2 / sigma^2)
    All kernels come from scikit-learn. The Cauchy kernel is the rational
    quadratic kernel with alpha = 1 and length scale l = sigma / sqrt(2),
    because (1 + d^2 / (2 l^2))^-1 = (1 + d^2 / sigma^2)^-1.
    """
    if kind == "linear":
        return linear_kernel(Z, Y)
    if kind == "poly":
        return polynomial_kernel(Z, Y, degree=degree, gamma=gamma, coef0=coef0)
    if kind == "rbf":
        return rbf_kernel(Z, Y, gamma=1 / (2 * sigma ** 2))
    if kind == "cauchy":
        return RationalQuadratic(length_scale=sigma / np.sqrt(2), alpha=1.0)(Z, Y)
    raise ValueError(f"unknown kernel {kind!r}")


def kpca(K):
    """Kernel PCA of a precomputed (uncentred) training kernel matrix K (n x n).

    scikit-learn centres K, eigendecomposes the centred matrix Kc and returns
    scores = eigenvector * sqrt(eigenvalue). Components with zero eigenvalue
    are dropped. Returns scores (n x k), eigenvalues (k,), the explained
    variance ratio lambda_k / trace(Kc) (share of the feature-space variance),
    and the fitted model (to project new samples later).
    """
    model = KernelPCA(kernel="precomputed")
    scores = model.fit_transform(K)
    lam = model.eigenvalues_
    evr = lam / np.trace(KernelCenterer().fit_transform(K))
    return scores, lam, evr, model


# ---------------------------------------------------------------------------
# Choosing kernel settings: cross-validated balanced kNN accuracy
# ---------------------------------------------------------------------------
def cv_balanced_accuracy(X, y, kind, factor=None, degree=None, n_comp=(2, 3, 7),
                         n_splits=5, n_repeats=10, k=5, seed=0):
    """Balanced kNN accuracy on the first q kPC scores, by repeated stratified CV.

    In every fold, everything is fitted on the training samples only:
    z-scoring, the width sigma = factor * median distance (rbf / cauchy) and
    kPCA. Test samples are scaled with the training mean/std and projected
    with their kernel values against the training samples.

    Returns {q: fold scores (n_splits * n_repeats,)} and
    {q: confusion matrix summed over all test folds}.
    """
    labels = np.unique(y)
    scores = {q: [] for q in n_comp}
    conf = {q: np.zeros((len(labels), len(labels)), dtype=int) for q in n_comp}
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    for tr, te in folds.split(X, y):
        Z_tr, mu, sd = zscore(X[tr])
        Z_te = (X[te] - mu) / sd
        sigma = factor * median_distance(Z_tr) if kind in ("rbf", "cauchy") else None
        args = dict(kind=kind, sigma=sigma, degree=degree)

        model = KernelPCA(kernel="precomputed", n_components=max(n_comp))
        T_tr = model.fit_transform(kernel_matrix(Z_tr, **args))
        T_te = model.transform(kernel_matrix(Z_te, Y=Z_tr, **args))   # test x train kernel

        for q in n_comp:
            knn = KNeighborsClassifier(n_neighbors=k).fit(T_tr[:, :q], y[tr])
            pred = knn.predict(T_te[:, :q])
            scores[q].append(balanced_accuracy_score(y[te], pred))
            conf[q] += confusion_matrix(y[te], pred, labels=labels)

    return {q: np.array(v) for q, v in scores.items()}, conf
