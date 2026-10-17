"""Shared functions for the Meter B kPCA project.

Each notebook imports this module; it grows one milestone at a time.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
