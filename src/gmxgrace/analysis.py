"""Scientific data transforms: GROMACS unit conversion, smoothing, replicate aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from gmxgrace.models import SeriesData

# Multiplicative conversion factors: (from_unit, to_unit) -> scale
_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("ps", "ns"): 1e-3,
    ("ps", "us"): 1e-6,
    ("ns", "ps"): 1e3,
    ("ns", "us"): 1e-3,
    ("us", "ns"): 1e3,
    ("us", "ps"): 1e6,
    ("nm", "A"): 10.0,
    ("A", "nm"): 0.1,
    ("nm2", "A2"): 100.0,
    ("A2", "nm2"): 0.01,
    ("kJ/mol", "kcal/mol"): 1.0 / 4.184,
    ("kcal/mol", "kJ/mol"): 4.184,
}

_UNIT_DISPLAY: dict[str, str] = {
    "ps": "ps",
    "ns": "ns",
    "us": "µs",
    "nm": "nm",
    "A": "Å",
    "nm2": "nm²",
    "A2": "Å²",
    "kJ/mol": "kJ mol⁻¹",
    "kcal/mol": "kcal mol⁻¹",
}


def unit_scale(from_unit: str | None, to_unit: str | None) -> float:
    """Return multiplicative factor from `from_unit` to `to_unit`, or 1.0 if unchanged/incompatible."""
    if not from_unit or not to_unit or to_unit == "original" or from_unit == to_unit:
        return 1.0
    return _UNIT_CONVERSIONS.get((from_unit, to_unit), 1.0)


def update_label_unit(label: str, from_unit: str | None, to_unit: str | None) -> str:
    """Replace unit text inside an axis label after unit conversion."""
    if not to_unit or to_unit == "original" or from_unit == to_unit:
        return label
    if unit_scale(from_unit, to_unit) == 1.0:
        return label
    disp = _UNIT_DISPLAY.get(to_unit, to_unit)
    # Replace parenthesized unit e.g. "(ps)" -> "(ns)", "(nm)" -> "(Å)"
    patterns = [
        r"\(\s*(?:ps|ns|us|µs|μs)\s*\)" if to_unit in {"ps", "ns", "us"} else None,
        r"\(\s*(?:nm|Å|A)\s*\)" if to_unit in {"nm", "A"} else None,
        r"\(\s*(?:nm²|nm\^2|Å²|A²|A\^2)\s*\)" if to_unit in {"nm2", "A2"} else None,
        r"\(\s*(?:kJ/mol|kcal/mol|kJ\s*mol[-⁻¹]*|kcal\s*mol[-⁻¹]*)\s*\)" if to_unit in {"kJ/mol", "kcal/mol"} else None,
    ]
    for pat in patterns:
        if pat and re.search(pat, label, flags=re.IGNORECASE):
            return re.sub(pat, f"({disp})", label, count=1, flags=re.IGNORECASE)
    return f"{label} ({disp})" if label else disp


def smooth_series(
    y: np.ndarray,
    method: str = "moving_avg",
    window: int = 25,
) -> np.ndarray:
    """Smooth a 1D array using moving average, Savitzky-Golay, or Gaussian filter."""
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    if method == "none" or n < 5:
        return arr.copy()

    w = max(3, min(int(window), n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w = max(3, w - 1)

    if method == "moving_avg":
        kernel = np.ones(w, dtype=float) / w
        pad = w // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:n]
    if method == "savgol":
        poly = min(3, w - 1)
        return savgol_filter(arr, window_length=w, polyorder=poly, mode="interp")
    if method == "gaussian":
        sigma = max(1.0, w / 4.0)
        return gaussian_filter1d(arr, sigma=sigma, mode="nearest")
    return arr.copy()


@dataclass
class AggregateResult:
    """Aggregated replicate curves (central tendency + lower/upper error envelope)."""

    x: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_replicates: int
    method: str


def aggregate_replicates(
    series_list: list[SeriesData],
    method: str = "mean_sd",
) -> AggregateResult | None:
    """Align multiple replicate series on their overlapping x-grid and compute mean±SD/SEM or median+IQR."""
    if not series_list or method == "none":
        return None
    if len(series_list) == 1:
        s = series_list[0]
        return AggregateResult(
            x=s.x.copy(),
            center=s.y.copy(),
            lower=s.y.copy(),
            upper=s.y.copy(),
            n_replicates=1,
            method=method,
        )

    # Check if all series share the exact same x grid
    same_grid = all(
        len(s.x) == len(series_list[0].x) and np.allclose(s.x, series_list[0].x, rtol=1e-5, atol=1e-8)
        for s in series_list[1:]
    )
    if same_grid:
        x_common = series_list[0].x.copy()
        mat = np.vstack([s.y for s in series_list])
    else:
        x_start = max(float(np.nanmin(s.x)) for s in series_list)
        x_end = min(float(np.nanmax(s.x)) for s in series_list)
        if x_end <= x_start:
            x_start = min(float(np.nanmin(s.x)) for s in series_list)
            x_end = max(float(np.nanmax(s.x)) for s in series_list)
        n_pts = int(np.median([len(s.x) for s in series_list]))
        x_common = np.linspace(x_start, x_end, max(10, n_pts))
        aligned = []
        for s in series_list:
            order = np.argsort(s.x)
            aligned.append(np.interp(x_common, s.x[order], s.y[order]))
        mat = np.vstack(aligned)

    n_rep = mat.shape[0]
    if method == "mean_sd":
        center = np.nanmean(mat, axis=0)
        spread = np.nanstd(mat, axis=0, ddof=1)
        lower, upper = center - spread, center + spread
    elif method == "mean_sem":
        center = np.nanmean(mat, axis=0)
        spread = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_rep)
        lower, upper = center - spread, center + spread
    elif method == "median_iqr":
        center = np.nanmedian(mat, axis=0)
        lower = np.nanpercentile(mat, 25, axis=0)
        upper = np.nanpercentile(mat, 75, axis=0)
    else:
        return None

    return AggregateResult(
        x=x_common,
        center=center,
        lower=lower,
        upper=upper,
        n_replicates=n_rep,
        method=method,
    )
