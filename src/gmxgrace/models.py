"""Core data structures and Pydantic figure specification models for gmxgrace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class SeriesData:
    """A single 1D/XY data series parsed from a GROMACS or tabular file."""

    name: str
    x: np.ndarray
    y: np.ndarray
    dy: np.ndarray | None = None
    dx: np.ndarray | None = None
    source_file: str = ""
    series_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_stats(self, eq_fraction: float = 0.0) -> dict[str, float | int | str]:
        """Compute summary statistics, optionally discarding an equilibration prefix."""
        y = np.asarray(self.y, dtype=float)
        x = np.asarray(self.x, dtype=float)
        valid = np.isfinite(y) & np.isfinite(x)
        y_v = y[valid]
        x_v = x[valid]
        if len(y_v) == 0:
            return {
                "name": self.name,
                "count": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "eq_mean": float("nan"),
                "eq_std": float("nan"),
            }
        start_idx = int(len(y_v) * min(max(eq_fraction, 0.0), 0.95))
        y_eq = y_v[start_idx:] if start_idx < len(y_v) else y_v
        return {
            "name": self.name,
            "count": int(len(y_v)),
            "x_min": float(np.min(x_v)),
            "x_max": float(np.max(x_v)),
            "min": float(np.min(y_v)),
            "max": float(np.max(y_v)),
            "mean": float(np.mean(y_v)),
            "std": float(np.std(y_v, ddof=1)) if len(y_v) > 1 else 0.0,
            "median": float(np.median(y_v)),
            "eq_mean": float(np.mean(y_eq)),
            "eq_std": float(np.std(y_eq, ddof=1)) if len(y_eq) > 1 else 0.0,
        }


@dataclass
class MatrixData:
    """A 2D matrix parsed from a GROMACS .xpm file (e.g., FEL, DSSP, contact map)."""

    values: np.ndarray  # shape (ny, nx), float for continuous or int codes for discrete
    x_axis: np.ndarray  # shape (nx,)
    y_axis: np.ndarray  # shape (ny,)
    is_discrete: bool = False
    categories: list[str] = field(default_factory=list)
    category_colors: list[str] = field(default_factory=list)
    legend_label: str = ""
    raw_colors: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedDataset:
    """Unified parsed representation for .xvg, .xpm, .edr, and tabular files."""

    filename: str
    file_format: Literal["xvg", "xpm", "edr", "tabular"]
    kind: Literal["xy", "matrix"] = "xy"
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    x_unit: str | None = None
    y_unit: str | None = None
    detected_type: str = "generic"  # rmsd, rmsf, gyrate, sasa, hbond, energy, pmf, fel, dssp, etc.
    gmx_command: str = ""
    series: list[SeriesData] = field(default_factory=list)
    matrix: MatrixData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self, eq_fraction: float = 0.0) -> dict[str, Any]:
        """Serialize metadata and stats (without huge arrays) for API & CLI inspect."""
        out: dict[str, Any] = {
            "filename": self.filename,
            "file_format": self.file_format,
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "detected_type": self.detected_type,
            "gmx_command": self.gmx_command,
        }
        if self.kind == "xy":
            out["series"] = [
                {
                    "index": s.series_index,
                    "name": s.name,
                    "has_dy": s.dy is not None,
                    "has_dx": s.dx is not None,
                    "stats": s.summary_stats(eq_fraction=eq_fraction),
                }
                for s in self.series
            ]
        elif self.matrix is not None:
            m = self.matrix
            out["matrix"] = {
                "shape": list(m.values.shape),
                "is_discrete": m.is_discrete,
                "legend_label": m.legend_label,
                "categories": m.categories,
                "category_colors": m.category_colors,
                "x_range": [float(np.min(m.x_axis)), float(np.max(m.x_axis))] if len(m.x_axis) else [0.0, 0.0],
                "y_range": [float(np.min(m.y_axis)), float(np.max(m.y_axis))] if len(m.y_axis) else [0.0, 0.0],
                "v_min": float(np.nanmin(m.values)) if m.values.size else 0.0,
                "v_max": float(np.nanmax(m.values)) if m.values.size else 0.0,
            }
        return out


# ---------------------------------------------------------------------------
# Declarative Pydantic Figure Specification (shared across CLI, API, Web UI)
# ---------------------------------------------------------------------------


class SeriesStyleSpec(BaseModel):
    """Per-series visual overrides."""

    dataset_id: str
    series_index: int = 0
    label: str | None = None
    visible: bool = True
    color: str | None = None
    linewidth: float | None = None
    linestyle: Literal["-", "--", "-.", ":", "none"] = "-"
    marker: Literal["none", "o", "s", "^", "D", "v", "x", "+"] = "none"
    markersize: float = 3.5
    alpha: float = 1.0
    fill_under: bool = False
    y_axis: Literal["left", "right"] = "left"
    zorder: int | None = None


class DomainTrackSpec(BaseModel):
    """Colored protein domain / structural motif bar rendered along the top of a panel (e.g. RMSF)."""

    start: float
    end: float
    label: str
    color: str = "#3B82F6"


class ZoomInsetSpec(BaseModel):
    """Magnified zoom inset box with publication connector lines."""

    enabled: bool = False
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    loc: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right"
    width_pct: float = Field(default=36.0, ge=15.0, le=65.0)
    height_pct: float = Field(default=34.0, ge=15.0, le=65.0)


class AnnotationSpec(BaseModel):
    """Reference lines, shaded regions, or text callouts on a panel."""

    kind: Literal["hline", "vline", "hspan", "vspan", "text"]
    value: float = 0.0
    value_end: float | None = None  # for hspan/vspan or y-coord for text
    text: str = ""
    color: str = "#555555"
    linestyle: Literal["-", "--", "-.", ":"] = "--"
    linewidth: float = 0.9
    alpha: float = 0.75


class PanelSpec(BaseModel):
    """Configuration for a single subplot panel."""

    id: str = "panel-1"
    title: str = ""
    subtitle: str = ""
    dataset_ids: list[str] = Field(default_factory=list)
    series_styles: list[SeriesStyleSpec] = Field(default_factory=list)
    plot_type: Literal[
        "auto",
        "line",
        "scatter",
        "density2d",
        "fel2d",
        "violin",
        "box",
        "delta",
        "convergence",
        "bar",
        "hist",
        "matrix",
        "contour",
    ] = "auto"

    # Labels & limits
    x_label: str | None = None
    y_label: str | None = None
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    x_scale: Literal["linear", "log"] = "linear"
    y_scale: Literal["linear", "log"] = "linear"

    # Dual Y-axis (Twin-Y)
    y2_label: str | None = None
    y2_min: float | None = None
    y2_max: float | None = None
    y2_unit: Literal["original", "nm", "A", "nm2", "A2", "kJ/mol", "kcal/mol"] = "original"

    # Unit conversions ("auto" / None means keep original)
    x_unit: Literal["original", "ps", "ns", "us", "nm", "A"] = "original"
    y_unit: Literal["original", "nm", "A", "nm2", "A2", "kJ/mol", "kcal/mol"] = "original"

    # Analysis & MD enhancements
    smoothing_method: Literal["none", "moving_avg", "savgol", "gaussian"] = "none"
    smoothing_window: int = Field(default=25, ge=3, le=2001)
    show_raw_under_smooth: bool = True
    raw_alpha: float = Field(default=0.25, ge=0.05, le=1.0)

    aggregate: Literal["none", "mean_sd", "mean_sem", "median_iqr"] = "none"
    aggregate_label: str = "Mean ± SD"
    aggregate_show_individual: bool = False
    aggregate_color: str | None = None

    show_errorbars: bool = True
    errorbar_style: Literal["band", "bars"] = "band"

    marginal_hist: bool = False
    marginal_kde: bool = True
    marginal_bins: int = Field(default=35, ge=5, le=200)

    equilibration_fraction: float = Field(default=0.0, ge=0.0, le=0.9)
    show_mean_line: bool = False

    # 2D FEL from two .xvg files
    fel_temp_k: float = Field(default=300.0, ge=10.0, le=2000.0)
    fel_bins: int = Field(default=45, ge=10, le=200)
    fel_unit: Literal["kJ/mol", "kcal/mol"] = "kJ/mol"
    fel_mark_minima: bool = True

    # System comparison stats (violin / box) & convergence
    show_pvalues: bool = True
    stat_test: Literal["mannwhitney", "welch"] = "mannwhitney"
    convergence_blocks: int = Field(default=5, ge=2, le=20)

    # Protein domain tracks & zoom inset
    domain_tracks: list[DomainTrackSpec] = Field(default_factory=list)
    zoom_inset: ZoomInsetSpec | None = None

    # Matrix (.xpm) specific options
    cmap: str = "RdYlBu_r"
    show_contours: bool = True
    contour_levels: int = Field(default=10, ge=2, le=50)
    colorbar_label: str | None = None
    v_min: float | None = None
    v_max: float | None = None
    matrix_interpolation: Literal["nearest", "bilinear", "bicubic", "gaussian"] = "nearest"

    # Legend & grid
    show_legend: bool = True
    legend_loc: str = "best"
    legend_ncol: int = Field(default=1, ge=1, le=6)
    legend_frame: bool = False
    grid: bool = False
    minor_ticks: bool = True

    annotations: list[AnnotationSpec] = Field(default_factory=list)


class FigureSpec(BaseModel):
    """Complete declarative publication figure specification."""

    journal: Literal["nature", "science", "acs", "biophysj", "pnas", "grace"] = "nature"
    column_width: Literal["single", "one_half", "double", "custom"] = "single"
    width_mm: float | None = None
    height_mm: float | None = None
    aspect_ratio: float = Field(default=0.72, ge=0.25, le=2.5)

    palette: str = "okabe-ito"
    font_family: Literal["sans-serif", "serif", "monospace"] | None = None
    font_size: float | None = None
    line_width: float | None = None

    nrows: int = Field(default=1, ge=1, le=6)
    ncols: int = Field(default=1, ge=1, le=6)
    share_x: bool = False
    share_y: bool = False
    panel_labels: Literal["none", "lowercase", "uppercase", "parens_lower", "parens_upper"] = "lowercase"
    suptitle: str = ""
    dpi: int = Field(default=300, ge=72, le=1200)
    transparent: bool = False
    max_plot_points: int = Field(default=12000, ge=0, le=500000)

    panels: list[PanelSpec] = Field(default_factory=lambda: [PanelSpec()])
