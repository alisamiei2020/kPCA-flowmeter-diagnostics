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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
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
def operating_point_groups(X, y, tol=0.1):
    """Group number for every row: consecutive rows of the same class whose average flow
    velocity differs by at most `tol` (m/s) belong to the same operating point.

    Rows in the data file are blocks of repeated test points at constant flow velocity
    (notebook 04), so neighbouring rows are near-duplicates and must not be split
    between training and test.
    """
    v = X[:, FEATURE_NAMES.index("Average flow velocity")]
    groups = np.zeros(len(y), dtype=int)
    for i in range(1, len(y)):
        new = y[i] != y[i - 1] or abs(v[i] - v[i - 1]) > tol
        groups[i] = groups[i - 1] + new
    return groups


def cv_splits(y, groups=None, n_splits=5, n_repeats=10, seed=0):
    """Train/test index pairs of repeated stratified K-fold CV.

    With `groups`, whole groups go to either training or test (StratifiedGroupKFold,
    reshuffled with seed + r in repeat r); otherwise ordinary RepeatedStratifiedKFold.
    """
    if groups is None:
        yield from RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                           random_state=seed).split(y, y)
        return
    for r in range(n_repeats):
        folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        yield from folds.split(y, y, groups)


def cv_balanced_accuracy(X, y, kind, factor=None, degree=None, n_comp=(2, 3, 7),
                         n_splits=5, n_repeats=10, k=5, seed=0, groups=None):
    """Balanced kNN accuracy on the first q kPC scores, by repeated stratified CV.

    In every fold, everything is fitted on the training samples only:
    z-scoring, the width sigma = factor * median distance (rbf / cauchy) and
    kPCA. Test samples are scaled with the training mean/std and projected
    with their kernel values against the training samples.

    With `groups`, whole groups are kept together (grouped CV, see cv_splits).

    Returns {q: split scores (n_splits * n_repeats,)}, {q: confusion matrix summed
    over all test folds} and the number of wrong predictions per sample (for q = n_comp[0]).
    """
    labels = np.unique(y)
    scores = {q: [] for q in n_comp}
    conf = {q: np.zeros((len(labels), len(labels)), dtype=int) for q in n_comp}
    wrong = np.zeros(len(y), dtype=int)

    for tr, te in cv_splits(y, groups, n_splits, n_repeats, seed):
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
            if q == n_comp[0]:
                wrong[te] += pred != y[te]

    return {q: np.array(v) for q, v in scores.items()}, conf, wrong


# ---------------------------------------------------------------------------
# Kernel MSPC: kPCA of healthy data, Hotelling T2 and SPE with control limits
# ---------------------------------------------------------------------------
def kmspc_fit(Zh, kind, sigma=None, degree=None, n_comp=4):
    """Fit a monitoring model on healthy (autoscaled) training samples Zh (n x p).

    kPCA of the centred kernel matrix Kc = U diag(lam) U'; training scores
    t_ik = sqrt(lam_k) u_ik. For every number of components H = 1..n_comp the
    control limits are mean + 3 std of the training T2 and SPE (as in the
    k-mspc toolbox).
    """
    K = kernel_matrix(Zh, kind, sigma=sigma, degree=degree)
    col_mean, all_mean = K.mean(axis=0), K.mean()
    Kc = K - col_mean[None, :] - col_mean[:, None] + all_mean        # centring in feature space
    lam, U = np.linalg.eigh(Kc)
    lam, U = lam[::-1][:n_comp], U[:, ::-1][:, :n_comp]               # largest first
    model = dict(Zh=Zh, kind=kind, sigma=sigma, degree=degree, U=U, lam=lam,
                 col_mean=col_mean, all_mean=all_mean, var=lam / (len(Zh) - 1))
    model["limits"] = {}
    for H in range(1, n_comp + 1):
        t2, spe = kmspc_stats(model, Zh, H)
        model["limits"][H] = (t2.mean() + 3 * t2.std(ddof=1), spe.mean() + 3 * spe.std(ddof=1))
    return model


def kmspc_stats(model, Z, H):
    """Hotelling T2 and SPE of samples Z (m x p) for a model from kmspc_fit, using H kPCs.

    t_k(x) = k_c(x)' u_k / sqrt(lam_k)                   (projection on kPC k)
    T2(x)  = sum_k t_k(x)^2 / var(t_k)                   (var(t_k) = lam_k / (n - 1))
    SPE(x) = k_c(x, x) - sum_k t_k(x)^2                  (squared distance to the kPC subspace)
    """
    args = dict(kind=model["kind"], sigma=model["sigma"], degree=model["degree"])
    Kx = kernel_matrix(Z, Y=model["Zh"], **args)                       # m x n
    row_mean = Kx.mean(axis=1)
    Kxc = Kx - row_mean[:, None] - model["col_mean"][None, :] + model["all_mean"]
    t = Kxc @ model["U"][:, :H] / np.sqrt(model["lam"][:H])
    kxx = np.diag(kernel_matrix(Z, **args))
    kxx_c = kxx - 2 * row_mean + model["all_mean"]
    t2 = (t ** 2 / model["var"][:H]).sum(axis=1)
    spe = np.maximum(kxx_c - (t ** 2).sum(axis=1), 0.0)
    return t2, spe


def healthy_scaling(X, healthy_rows):
    """Columns that vary among the healthy training rows, and their healthy mean and std."""
    keep = np.ptp(X[healthy_rows], axis=0) > 0
    mu = X[healthy_rows][:, keep].mean(axis=0)
    sd = X[healthy_rows][:, keep].std(axis=0, ddof=1)
    return keep, mu, sd


def kmspc_model(X, healthy_rows, kind, factor=None, degree=None, H_max=4):
    """Monitoring model from raw data X and the indices of the healthy training rows:
    drop variables constant among them, scale with their mean/std, set
    sigma = factor * median distance (rbf / cauchy) and fit kmspc_fit."""
    keep, mu, sd = healthy_scaling(X, healthy_rows)
    Zh = (X[healthy_rows][:, keep] - mu) / sd
    sigma = factor * median_distance(Zh) if kind in ("rbf", "cauchy") else None
    model = kmspc_fit(Zh, kind, sigma=sigma, degree=degree, n_comp=H_max)
    model.update(keep=keep, mu=mu, sd=sd)
    return model


def kmspc_monitor(model, X, rows, H):
    """T2 and SPE of raw rows X[rows] under a model from kmspc_model, with H kPCs."""
    Z = (X[rows][:, model["keep"]] - model["mu"]) / model["sd"]
    return kmspc_stats(model, Z, H)


def oos_limits(X, healthy_rows, groups, kind, factor=None, degree=None, H_max=4):
    """Out-of-sample control limits: every healthy operating-point group among
    `healthy_rows` is left out once, the model is fitted on the others, and T2 / SPE
    of the left-out rows are pooled. Limits = mean + 3 std of the pooled values.

    The training-sample limits of kmspc_fit are too tight when there are few healthy
    samples, because a model always fits its own training samples better than new ones.
    """
    pooled = {H: ([], []) for H in range(1, H_max + 1)}
    for g in np.unique(groups[healthy_rows]):
        fit_rows = healthy_rows[groups[healthy_rows] != g]
        out_rows = healthy_rows[groups[healthy_rows] == g]
        model = kmspc_model(X, fit_rows, kind, factor, degree, H_max)
        for H in pooled:
            t2, spe = kmspc_monitor(model, X, out_rows, H)
            pooled[H][0].extend(t2)
            pooled[H][1].extend(spe)
    return {H: (np.mean(t2) + 3 * np.std(t2, ddof=1), np.mean(spe) + 3 * np.std(spe, ddof=1))
            for H, (t2, spe) in pooled.items()}


def monitoring_rates(alarm, y_test):
    """False-alarm rate, detection rates and rho_b for one boolean alarm vector."""
    far = alarm[y_test == 1].mean()
    return dict(FAR=far, gas=alarm[y_test == 2].mean(), waxing=alarm[y_test == 3].mean(),
                rho=1 - ((1 - far) + alarm[y_test > 1].mean()) / 2)


def cv_monitoring(X, y, groups, kind, factor=None, degree=None, H_max=4, limits="oos",
                  n_splits=5, n_repeats=10, seed=0):
    """Grouped CV of kernel MSPC. In every fold the model and its limits use the healthy
    training rows only (limits="oos": leave-one-group-out limits; "train": training-sample
    limits as in the k-mspc toolbox); all test rows are then monitored.

    Returns {stat: {metric: array (n_splits * n_repeats, H_max)}} for stat in "T2", "SPE",
    "either" (alarm if T2 or SPE is above its limit) and metric in "FAR", "gas", "waxing",
    "rho" (rho_b = 1 - (share of healthy rows without alarm + share of faulty rows with
    alarm) / 2).
    """
    out = {s: {m: [] for m in ("FAR", "gas", "waxing", "rho")} for s in ("T2", "SPE", "either")}
    for tr, te in cv_splits(y, groups, n_splits, n_repeats, seed):
        h_tr = tr[y[tr] == 1]
        model = kmspc_model(X, h_tr, kind, factor, degree, H_max)
        lims = oos_limits(X, h_tr, groups, kind, factor, degree, H_max) if limits == "oos" else model["limits"]
        rows = {s: {m: [] for m in out[s]} for s in out}
        for H in range(1, H_max + 1):
            t2, spe = kmspc_monitor(model, X, te, H)
            alarms = {"T2": t2 > lims[H][0], "SPE": spe > lims[H][1]}
            alarms["either"] = alarms["T2"] | alarms["SPE"]
            for s, a in alarms.items():
                for m, val in monitoring_rates(a, y[te]).items():
                    rows[s][m].append(val)
        for s in out:
            for m in out[s]:
                out[s][m].append(rows[s][m])
    return {s: {m: np.array(v) for m, v in d.items()} for s, d in out.items()}
