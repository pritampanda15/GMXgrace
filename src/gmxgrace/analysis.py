"""Scientific data transforms: GROMACS unit conversion, smoothing, replicate aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu, ttest_ind

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


def minmax_downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 12000,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak-preserving vectorized Min-Max envelope downsampling for heavy MD trajectories.

    Preserves the exact visual minimum and maximum in every bin (in chronological order),
    keeping SVGs/PDFs lightweight and crisp even for 1M+ frame trajectories while never
    altering statistical calculations (which always run on full-resolution data).
    """
    n = len(x)
    if max_points <= 0 or n <= max_points or n < 10:
        return x, y

    nbins = max(2, (max_points - 2) // 2)
    interior = n - 2
    bin_size = interior // nbins
    if bin_size < 2:
        return x, y

    usable = nbins * bin_size
    y_slice = y[1 : 1 + usable].reshape(nbins, bin_size)
    # Replace NaNs safely for argmin/argmax
    has_nan = np.isnan(y_slice).any()
    if has_nan:
        y_min_safe = np.where(np.isnan(y_slice), np.inf, y_slice)
        y_max_safe = np.where(np.isnan(y_slice), -np.inf, y_slice)
        idx_min = np.argmin(y_min_safe, axis=1)
        idx_max = np.argmax(y_max_safe, axis=1)
    else:
        idx_min = np.argmin(y_slice, axis=1)
        idx_max = np.argmax(y_slice, axis=1)

    offsets = 1 + np.arange(nbins) * bin_size
    g_min = offsets + idx_min
    g_max = offsets + idx_max

    first = np.minimum(g_min, g_max)
    second = np.maximum(g_min, g_max)
    interleaved = np.empty(nbins * 2, dtype=np.intp)
    interleaved[0::2] = first
    interleaved[1::2] = second

    indices = np.concatenate(([0], interleaved, [n - 1]))
    return x[indices], y[indices]


def minmax_downsample_band(
    x: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a curve with lower/upper confidence envelope preserving outer band bounds."""
    n = len(x)
    if max_points <= 0 or n <= max_points or n < 10:
        return x, center, lower, upper

    nbins = max(4, max_points)
    bin_size = n // nbins
    if bin_size < 2:
        return x, center, lower, upper

    usable = nbins * bin_size
    x_b = x[:usable].reshape(nbins, bin_size).mean(axis=1)
    c_b = np.nanmean(center[:usable].reshape(nbins, bin_size), axis=1)
    lo_b = np.nanmin(lower[:usable].reshape(nbins, bin_size), axis=1)
    hi_b = np.nanmax(upper[:usable].reshape(nbins, bin_size), axis=1)
    return x_b, c_b, lo_b, hi_b


# ---------------------------------------------------------------------------
# Biophysics Analysis: 2D FEL from Two XVGs, Convergence, Pairwise Stats
# ---------------------------------------------------------------------------

_R_KJ_MOL_K = 8.314462618e-3  # kJ / (mol * K)
_R_KCAL_MOL_K = 1.987204259e-3  # kcal / (mol * K)


@dataclass
class FELMinimum:
    """Representative energy basin minimum mapped back to an actual MD frame."""

    rank: int
    x: float
    y: float
    dg: float
    time: float
    frame_index: int
    trjconv_cmd: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "rank": self.rank,
            "label": f"M{self.rank}",
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "dg": round(self.dg, 3),
            "time": round(self.time, 3),
            "frame_index": self.frame_index,
            "trjconv_cmd": self.trjconv_cmd,
        }


@dataclass
class FELResult:
    """Computed 2D Gibbs Free Energy Landscape ΔG(x, y) = -k_B T ln P(x, y)."""

    x_centers: np.ndarray
    y_centers: np.ndarray
    dg_grid: np.ndarray  # shape (ny, nx)
    unit: str
    temp_k: float
    minima: list[FELMinimum]
    x_vals: np.ndarray
    y_vals: np.ndarray
    t_vals: np.ndarray


def compute_fel_from_series(
    s_x: SeriesData,
    s_y: SeriesData,
    *,
    temp_k: float = 300.0,
    bins: int = 45,
    unit: str = "kJ/mol",
    eq_fraction: float = 0.0,
    max_minima: int = 3,
    time_unit: str | None = None,
) -> FELResult:
    """Compute a 2D Gibbs Free Energy Landscape from any two collective-variable series.

    Aligns both series by timestamp (or index if lengths match), computes ΔG = -RT ln(P / P_max),
    locates the global and distinct local minima, and maps each minimum back to the nearest
    real trajectory frame for `gmx trjconv -dump`.
    """
    tx, vx = np.asarray(s_x.x, dtype=float), np.asarray(s_x.y, dtype=float)
    ty, vy = np.asarray(s_y.x, dtype=float), np.asarray(s_y.y, dtype=float)

    if len(tx) == len(ty) and np.allclose(tx, ty, rtol=1e-4, atol=1e-6):
        t_common, x_aligned, y_aligned = tx, vx, vy
    else:
        t_start = max(float(np.nanmin(tx)), float(np.nanmin(ty)))
        t_end = min(float(np.nanmax(tx)), float(np.nanmax(ty)))
        mask = (tx >= t_start) & (tx <= t_end)
        t_common = tx[mask] if np.sum(mask) >= 10 else tx
        x_aligned = vx[mask] if np.sum(mask) >= 10 else vx
        order_y = np.argsort(ty)
        y_aligned = np.interp(t_common, ty[order_y], vy[order_y])

    valid = np.isfinite(t_common) & np.isfinite(x_aligned) & np.isfinite(y_aligned)
    t_common, x_aligned, y_aligned = t_common[valid], x_aligned[valid], y_aligned[valid]

    start_idx = int(len(t_common) * min(max(eq_fraction, 0.0), 0.9))
    t_v = t_common[start_idx:]
    x_v = x_aligned[start_idx:]
    y_v = y_aligned[start_idx:]

    if len(x_v) < 10:
        raise ValueError("Need at least 10 aligned trajectory frames to compute a 2D FEL.")

    nb = max(10, min(int(bins), 150))
    hist, x_edges, y_edges = np.histogram2d(x_v, y_v, bins=nb)
    # Light Gaussian smoothing produces clean publication contours while preserving basin positions
    hist_smooth = gaussian_filter(hist.T.astype(float), sigma=1.15)  # shape (ny, nx)
    p_max = float(np.max(hist_smooth))
    if p_max <= 0:
        raise ValueError("Empty 2D histogram — cannot compute free energy landscape.")

    rt = (_R_KCAL_MOL_K if unit == "kcal/mol" else _R_KJ_MOL_K) * float(temp_k)
    prob = hist_smooth / p_max
    with np.errstate(divide="ignore"):
        dg = np.where(prob > 1e-6, -rt * np.log(np.maximum(prob, 1e-6)), np.nan)
    finite_max = float(np.nanmax(dg)) if np.any(np.isfinite(dg)) else 15.0
    dg_filled = np.where(np.isfinite(dg), dg, finite_max * 1.05)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Locate distinct local energy minima using 2D neighborhood minimum filter
    neighborhood = max(3, nb // 8)
    local_min_mask = (dg_filled == minimum_filter(dg_filled, size=neighborhood)) & np.isfinite(dg)
    cand_rows, cand_cols = np.where(local_min_mask)
    candidates = sorted(
        [(float(dg_filled[r, c]), int(r), int(c)) for r, c in zip(cand_rows, cand_cols)],
        key=lambda item: item[0],
    )

    # Filter out nearly overlapping basins (min separation 12% of axis span)
    x_span = max(float(x_centers[-1] - x_centers[0]), 1e-9)
    y_span = max(float(y_centers[-1] - y_centers[0]), 1e-9)
    selected: list[tuple[float, float, float]] = []
    for dg_val, r, c in candidates:
        xc, yc = float(x_centers[c]), float(y_centers[r])
        too_close = any(
            ((xc - sx) / x_span) ** 2 + ((yc - sy) / y_span) ** 2 < 0.14**2 for _, sx, sy in selected
        )
        if not too_close:
            selected.append((dg_val, xc, yc))
        if len(selected) >= max_minima:
            break

    # Map each basin back to the closest real frame in the trajectory
    x_std = max(float(np.std(x_v)), 1e-9)
    y_std = max(float(np.std(y_v)), 1e-9)
    tu_ps_mult = 1000.0 if time_unit == "ns" else (1e6 if time_unit == "us" else 1.0)

    minima: list[FELMinimum] = []
    for rank, (dg_val, xc, yc) in enumerate(selected, start=1):
        dist2 = ((x_v - xc) / x_std) ** 2 + ((y_v - yc) / y_std) ** 2
        best_i = int(np.argmin(dist2))
        t_frame = float(t_v[best_i])
        t_ps = t_frame * tu_ps_mult
        cmd = f"gmx trjconv -s md.tpr -f md.xtc -o basin_M{rank}.pdb -dump {t_ps:.1f}"
        minima.append(
            FELMinimum(
                rank=rank,
                x=float(x_v[best_i]),
                y=float(y_v[best_i]),
                dg=dg_val,
                time=t_frame,
                frame_index=start_idx + best_i,
                trjconv_cmd=cmd,
            )
        )

    return FELResult(
        x_centers=x_centers,
        y_centers=y_centers,
        dg_grid=dg_filled,
        unit=unit,
        temp_k=temp_k,
        minima=minima,
        x_vals=x_v,
        y_vals=y_v,
        t_vals=t_v,
    )


@dataclass
class ConvergenceMetrics:
    """Trajectory convergence & Flyvbjerg-Petersen block-averaging metrics."""

    block_centers: np.ndarray
    block_means: np.ndarray
    block_stds: np.ndarray
    block_sem: float
    drift_total: float
    drift_ratio: float  # |drift_total| / std
    tau_frames: float
    n_eff: float
    status: str  # "Converged" | "Mild Drift" | "Drifting"

    def to_summary_dict(self) -> dict[str, float | str]:
        return {
            "block_sem": round(self.block_sem, 5),
            "drift_total": round(self.drift_total, 5),
            "tau_frames": round(self.tau_frames, 2),
            "n_eff": round(self.n_eff, 1),
            "convergence_status": self.status,
        }


def compute_convergence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eq_fraction: float = 0.0,
    n_blocks: int = 5,
) -> ConvergenceMetrics:
    """Compute block-averaged means/SEM, linear drift, FFT autocorrelation time, and N_eff."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xv, yv = xa[valid], ya[valid]
    start = int(len(yv) * min(max(eq_fraction, 0.0), 0.9))
    xv, yv = xv[start:], yv[start:]

    n = len(yv)
    if n < 6:
        return ConvergenceMetrics(
            block_centers=xv[:1] if n else np.array([0.0]),
            block_means=yv[:1] if n else np.array([0.0]),
            block_stds=np.array([0.0]),
            block_sem=0.0,
            drift_total=0.0,
            drift_ratio=0.0,
            tau_frames=1.0,
            n_eff=float(n),
            status="Insufficient Data",
        )

    nb = max(2, min(int(n_blocks), n // 3))
    x_splits = np.array_split(xv, nb)
    y_splits = np.array_split(yv, nb)
    b_centers = np.array([float(np.mean(xs)) for xs in x_splits])
    b_means = np.array([float(np.mean(ys)) for ys in y_splits])
    b_stds = np.array([float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0 for ys in y_splits])
    block_sem = float(np.std(b_means, ddof=1) / np.sqrt(nb))

    # Linear drift across analyzed window
    x_span = float(xv[-1] - xv[0])
    if abs(x_span) > 1e-12:
        slope, _ = np.polyfit(xv - xv[0], yv, 1)
        drift_total = float(slope * x_span)
    else:
        drift_total = 0.0
    y_std = float(np.std(yv, ddof=1))
    drift_ratio = abs(drift_total) / max(y_std, 1e-12)

    # Fast O(N log N) FFT autocorrelation time (initial positive sequence)
    yc = yv - np.mean(yv)
    var = float(np.dot(yc, yc))
    if var <= 1e-24:
        tau_frames = 1.0
    else:
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        fft = np.fft.rfft(yc, n=nfft)
        acf = np.fft.irfft(fft * np.conjugate(fft), n=nfft)[:n] / var
        # Sum positive initial sequence up to first zero crossing (capped at n // 4)
        neg = np.where(acf[1 : max(2, n // 4)] <= 0)[0]
        cutoff = int(neg[0]) + 1 if len(neg) else max(2, n // 4)
        tau_frames = float(max(0.5, 0.5 + np.sum(acf[1:cutoff])))

    g = max(1.0, 1.0 + 2.0 * tau_frames)
    n_eff = float(max(1.0, n / g))

    if drift_ratio < 0.65:
        status = "Converged"
    elif drift_ratio < 1.35:
        status = "Mild Drift"
    else:
        status = "Drifting"

    return ConvergenceMetrics(
        block_centers=b_centers,
        block_means=b_means,
        block_stds=b_stds,
        block_sem=block_sem,
        drift_total=drift_total,
        drift_ratio=drift_ratio,
        tau_frames=tau_frames,
        n_eff=n_eff,
        status=status,
    )


def pvalue_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


@dataclass
class PairwiseTestResult:
    group_a: int
    group_b: int
    label_a: str
    label_b: str
    p_value: float
    stars: str
    method: str


def compute_pairwise_tests(
    groups: list[tuple[str, np.ndarray]],
    method: str = "mannwhitney",
) -> list[PairwiseTestResult]:
    """Compute pairwise Mann-Whitney U or Welch t-tests between equilibrated MD groups."""
    results: list[PairwiseTestResult] = []
    n = len(groups)
    # Cap pairwise brackets cleanly so plots with many groups stay uncluttered
    pairs: list[tuple[int, int]] = []
    if n <= 4:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        # Compare every system against group 0 (control/apo)
        for j in range(1, min(n, 6)):
            pairs.append((0, j))

    for i, j in pairs:
        la, a = groups[i]
        lb, b = groups[j]
        av = a[np.isfinite(a)]
        bv = b[np.isfinite(b)]
        if len(av) < 3 or len(bv) < 3:
            continue
        try:
            if method == "welch":
                _, pval = ttest_ind(av, bv, equal_var=False)
            else:
                _, pval = mannwhitneyu(av, bv, alternative="two-sided")
            p_float = float(pval)
        except Exception:
            p_float = float("nan")
        results.append(
            PairwiseTestResult(
                group_a=i,
                group_b=j,
                label_a=la,
                label_b=lb,
                p_value=p_float,
                stars=pvalue_to_stars(p_float),
                method=method,
            )
        )
    return results
