"""Shared functions for the Meter B kPCA project.

Each notebook imports this module; it grows one milestone at a time.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from sklearn.decomposition import PCA

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


def biplot(ax, T, A, y, names, pcs=(0, 1), n_arrows=15, circle=False, evr=None):
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
    lab = [f"PC{p + 1}" + (f" ({evr[p]:.1%})" if evr is not None else "") for p in pcs]
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    return scale
