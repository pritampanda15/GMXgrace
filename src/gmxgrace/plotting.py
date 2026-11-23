"""Unified publication-quality Matplotlib rendering engine for CLI, Python API, and Web Studio."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.stats import gaussian_kde

from gmxgrace.analysis import aggregate_replicates, smooth_series, unit_scale, update_label_unit
from gmxgrace.models import (
    AnnotationSpec,
    FigureSpec,
    PanelSpec,
    ParsedDataset,
    SeriesData,
    SeriesStyleSpec,
)
from gmxgrace.styles import MM_PER_INCH, build_rc_params, get_journal_preset, get_palette

# Matplotlib state lock for safe concurrent FastAPI requests
_RENDER_LOCK = threading.Lock()


def _format_panel_label(index: int, style: str) -> str | None:
    if style == "none":
        return None
    letter = chr(ord("a") + (index % 26))
    if style == "lowercase":
        return letter
    if style == "uppercase":
        return letter.upper()
    if style == "parens_lower":
        return f"({letter})"
    if style == "parens_upper":
        return f"({letter.upper()})"
    return letter


def _compute_figure_size_inches(spec: FigureSpec) -> tuple[float, float]:
    preset = get_journal_preset(spec.journal)
    if spec.column_width == "custom" and spec.width_mm:
        w_mm = spec.width_mm
    else:
        w_mm = preset.widths_mm.get(spec.column_width, preset.widths_mm["single"])
        # For multi-column grids, ensure single-column width scales sensibly if user didn't pick double
        if spec.ncols >= 2 and spec.column_width == "single" and spec.width_mm is None:
            w_mm = preset.widths_mm["double"]

    if spec.height_mm:
        h_mm = spec.height_mm
    else:
        cell_w = w_mm / max(1, spec.ncols)
        cell_h = cell_w * spec.aspect_ratio
        h_mm = cell_h * max(1, spec.nrows)

    return (max(w_mm / MM_PER_INCH, 2.0), max(h_mm / MM_PER_INCH, 1.6))


def _resolve_panel_series(
    panel: PanelSpec,
    datasets: dict[str, ParsedDataset],
) -> tuple[list[tuple[SeriesData, SeriesStyleSpec | None, ParsedDataset]], ParsedDataset | None]:
    """Collect active XY series (or a matrix dataset) belonging to a panel."""
    style_lookup: dict[tuple[str, int], SeriesStyleSpec] = {
        (st.dataset_id, st.series_index): st for st in panel.series_styles
    }
    xy_items: list[tuple[SeriesData, SeriesStyleSpec | None, ParsedDataset]] = []
    matrix_ds: ParsedDataset | None = None

    for ds_id in panel.dataset_ids:
        ds = datasets.get(ds_id)
        if ds is None:
            continue
        if ds.kind == "matrix":
            if matrix_ds is None:
                matrix_ds = ds
            continue
        for s in ds.series:
            st = style_lookup.get((ds_id, s.series_index))
            if st is not None and not st.visible:
                continue
            xy_items.append((s, st, ds))
    return xy_items, matrix_ds


def _apply_annotations(ax: Axes, annotations: list[AnnotationSpec]) -> None:
    for ann in annotations:
        if ann.kind == "hline":
            ax.axhline(
                ann.value,
                color=ann.color,
                linestyle=ann.linestyle,
                linewidth=ann.linewidth,
                alpha=ann.alpha,
                zorder=5,
            )
            if ann.text:
                ax.text(
                    0.98,
                    ann.value,
                    f" {ann.text}",
                    color=ann.color,
                    fontsize=max(6.0, plt.rcParams["legend.fontsize"] * 0.92),
                    ha="right",
                    va="bottom",
                    transform=ax.get_yaxis_transform(),
                )
        elif ann.kind == "vline":
            ax.axvline(
                ann.value,
                color=ann.color,
                linestyle=ann.linestyle,
                linewidth=ann.linewidth,
                alpha=ann.alpha,
                zorder=5,
            )
            if ann.text:
                ax.text(
                    ann.value,
                    0.97,
                    f" {ann.text}",
                    color=ann.color,
                    fontsize=max(6.0, plt.rcParams["legend.fontsize"] * 0.92),
                    ha="left",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )
        elif ann.kind == "hspan" and ann.value_end is not None:
            ax.axhspan(ann.value, ann.value_end, color=ann.color, alpha=min(ann.alpha, 0.25), zorder=1)
        elif ann.kind == "vspan" and ann.value_end is not None:
            ax.axvspan(ann.value, ann.value_end, color=ann.color, alpha=min(ann.alpha, 0.25), zorder=1)
            if ann.text:
                mid = 0.5 * (ann.value + ann.value_end)
                ax.text(
                    mid,
                    0.96,
                    ann.text,
                    color=ann.color,
                    fontsize=max(6.0, plt.rcParams["legend.fontsize"] * 0.9),
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )
        elif ann.kind == "text" and ann.text:
            y_pos = ann.value_end if ann.value_end is not None else 0.5
            ax.text(
                ann.value,
                y_pos,
                ann.text,
                color=ann.color,
                fontsize=max(6.5, plt.rcParams["legend.fontsize"]),
                ha="left",
                va="bottom",
            )


def _render_matrix_panel(
    ax: Axes,
    panel: PanelSpec,
    ds: ParsedDataset,
    preset_axes_lw: float,
) -> None:
    assert ds.matrix is not None
    m = ds.matrix

    x_scale = unit_scale(ds.x_unit, panel.x_unit)
    y_scale = unit_scale(ds.y_unit, panel.y_unit)
    x_axis = m.x_axis * x_scale
    y_axis = m.y_axis * y_scale

    extent = [
        float(x_axis[0]) if len(x_axis) else 0.0,
        float(x_axis[-1]) if len(x_axis) else float(m.values.shape[1]),
        float(y_axis[0]) if len(y_axis) else 0.0,
        float(y_axis[-1]) if len(y_axis) else float(m.values.shape[0]),
    ]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)

    if m.is_discrete and m.categories:
        # Discrete secondary structure (DSSP) / categorical matrix
        colors = [c if c.startswith("#") else "#888888" for c in m.category_colors]
        # Replace pure white in DSSP coil with soft light gray so it's readable on white paper
        colors = ["#F1F5F9" if c.upper() in {"#FFFFFF", "#FFF"} else c for c in colors]
        cmap = ListedColormap(colors)
        n_cat = len(m.categories)
        bounds = np.arange(-0.5, n_cat + 0.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(
            m.values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        cbar = ax.figure.colorbar(im, cax=cax, ticks=np.arange(n_cat))
        cbar.ax.set_yticklabels(m.categories, fontsize=max(5.5, plt.rcParams["ytick.labelsize"] * 0.92))
        cbar.set_label(panel.colorbar_label or m.legend_label)
    else:
        vmin = panel.v_min if panel.v_min is not None else float(np.nanmin(m.values))
        vmax = panel.v_max if panel.v_max is not None else float(np.nanmax(m.values))
        if panel.plot_type == "contour":
            xx, yy = np.meshgrid(x_axis, y_axis)
            cf = ax.contourf(
                xx,
                yy,
                m.values,
                levels=panel.contour_levels,
                cmap=panel.cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if panel.show_contours:
                ax.contour(
                    xx,
                    yy,
                    m.values,
                    levels=panel.contour_levels,
                    colors="#1F2937",
                    linewidths=0.45,
                    alpha=0.55,
                )
            cbar = ax.figure.colorbar(cf, cax=cax)
        else:
            interp = panel.matrix_interpolation
            if interp == "nearest" and ds.detected_type == "fel":
                interp = "bilinear"
            im = ax.imshow(
                m.values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=panel.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interp,
            )
            if panel.show_contours and ds.detected_type != "dssp":
                xx, yy = np.meshgrid(x_axis, y_axis)
                try:
                    ax.contour(
                        xx,
                        yy,
                        m.values,
                        levels=panel.contour_levels,
                        colors="#111827",
                        linewidths=0.45,
                        alpha=0.5,
                    )
                except ValueError:
                    pass
            cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label(panel.colorbar_label or m.legend_label)

    cbar.outline.set_linewidth(preset_axes_lw)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)

    x_lbl = panel.x_label if panel.x_label is not None else update_label_unit(ds.x_label, ds.x_unit, panel.x_unit)
    y_lbl = panel.y_label if panel.y_label is not None else update_label_unit(ds.y_label, ds.y_unit, panel.y_unit)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    if panel.title:
        ax.set_title(panel.title, pad=5)
    elif ds.title:
        ax.set_title(ds.title, pad=5)

    if panel.x_min is not None or panel.x_max is not None:
        ax.set_xlim(panel.x_min, panel.x_max)
    if panel.y_min is not None or panel.y_max is not None:
        ax.set_ylim(panel.y_min, panel.y_max)

    _apply_annotations(ax, panel.annotations)


def _render_density2d(
    ax: Axes,
    panel: PanelSpec,
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]],
    first_ds: ParsedDataset,
) -> None:
    """Render a publication 2D KDE contour + scatter plot (Ramachandran phi/psi or PCA 2D)."""
    all_x = np.concatenate([t[0].x for t in transformed_series])
    all_y = np.concatenate([t[0].y for t in transformed_series])
    valid = np.isfinite(all_x) & np.isfinite(all_y)
    xv, yv = all_x[valid], all_y[valid]
    if len(xv) < 10:
        ax.scatter(xv, yv, s=8.0, color="#1D4ED8", alpha=0.7)
        return

    is_rama = first_ds.detected_type == "ramachandran"
    x_lo, x_hi = (-180.0, 180.0) if is_rama else (float(np.min(xv)), float(np.max(xv)))
    y_lo, y_hi = (-180.0, 180.0) if is_rama else (float(np.min(yv)), float(np.max(yv)))
    gx = np.linspace(x_lo, x_hi, 120)
    gy = np.linspace(y_lo, y_hi, 120)
    xx, yy = np.meshgrid(gx, gy)
    try:
        kde = gaussian_kde(np.vstack([xv, yv]))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        cf = ax.contourf(xx, yy, zz, levels=panel.contour_levels, cmap=panel.cmap)
        if panel.show_contours:
            ax.contour(xx, yy, zz, levels=panel.contour_levels, colors="#111827", linewidths=0.4, alpha=0.45)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.08)
        cbar = ax.figure.colorbar(cf, cax=cax)
        cbar.set_label(panel.colorbar_label or "Density")
    except Exception:
        pass
    ax.scatter(xv, yv, s=3.5, color="#111827", alpha=0.18, edgecolors="none", zorder=4)
    if is_rama:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
        ax.axhline(0, color="#64748B", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axvline(0, color="#64748B", linewidth=0.5, linestyle=":", alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


def _render_xy_panel(
    ax: Axes,
    panel: PanelSpec,
    xy_items: list[tuple[SeriesData, SeriesStyleSpec | None, ParsedDataset]],
    palette: list[str],
    default_lw: float,
) -> None:
    if not xy_items:
        ax.text(
            0.5,
            0.5,
            "Drop GROMACS files (.xvg, .xpm, .edr, .csv)\nonto this panel",
            ha="center",
            va="center",
            color="#94A3B8",
            fontsize=8.5,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    first_ds = xy_items[0][2]
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]] = []

    for idx, (s, st, ds) in enumerate(xy_items):
        xs = unit_scale(ds.x_unit, panel.x_unit)
        ys = unit_scale(ds.y_unit, panel.y_unit)
        x_arr = s.x * xs
        y_arr = s.y * ys
        dy_arr = s.dy * ys if s.dy is not None else None
        dx_arr = s.dx * xs if s.dx is not None else None
        color = (st.color if st and st.color else None) or palette[idx % len(palette)]
        label = (st.label if st and st.label else None) or s.name

        transformed_series.append(
            (
                SeriesData(
                    name=label,
                    x=x_arr,
                    y=y_arr,
                    dy=dy_arr,
                    dx=dx_arr,
                    source_file=s.source_file,
                    series_index=s.series_index,
                ),
                st,
                color,
            )
        )

    all_y_for_marginal: list[tuple[np.ndarray, str]] = []

    # Replicate aggregation mode (e.g., 3 independent MD trajectories -> Mean ± SD band)
    if panel.aggregate != "none" and len(transformed_series) >= 2:
        if panel.aggregate_show_individual:
            for s_t, st, col in transformed_series:
                ax.plot(
                    s_t.x,
                    s_t.y,
                    color=col,
                    linewidth=max(0.5, default_lw * 0.55),
                    alpha=0.28,
                    zorder=2,
                )
        agg = aggregate_replicates([t[0] for t in transformed_series], method=panel.aggregate)
        if agg is not None:
            agg_col = panel.aggregate_color or palette[0]
            center = agg.center
            lower = agg.lower
            upper = agg.upper
            if panel.smoothing_method != "none":
                center = smooth_series(center, panel.smoothing_method, panel.smoothing_window)
                lower = smooth_series(lower, panel.smoothing_method, panel.smoothing_window)
                upper = smooth_series(upper, panel.smoothing_method, panel.smoothing_window)

            agg_lbl = panel.aggregate_label or {
                "mean_sd": f"Mean ± SD (n={agg.n_replicates})",
                "mean_sem": f"Mean ± SEM (n={agg.n_replicates})",
                "median_iqr": f"Median + IQR (n={agg.n_replicates})",
            }.get(panel.aggregate, "Aggregate")

            ax.fill_between(agg.x, lower, upper, color=agg_col, alpha=0.22, linewidth=0, zorder=3)
            ax.plot(agg.x, center, color=agg_col, linewidth=default_lw * 1.15, label=agg_lbl, zorder=4)
            all_y_for_marginal.append((center, agg_col))

            if panel.show_mean_line:
                start = int(len(center) * panel.equilibration_fraction)
                mean_val = float(np.nanmean(center[start:]))
                ax.axhline(mean_val, color=agg_col, linestyle="--", linewidth=0.8, alpha=0.8)
    else:
        # Standard multi-series plotting
        plot_kind = panel.plot_type
        if plot_kind == "auto":
            plot_kind = "density2d" if first_ds.detected_type == "ramachandran" else "line"

        if plot_kind == "density2d":
            _render_density2d(ax, panel, transformed_series, first_ds)
        for idx, (s_t, st, col) in enumerate([] if plot_kind == "density2d" else transformed_series):
            lw = (st.linewidth if st and st.linewidth is not None else default_lw)
            ls = st.linestyle if st else "-"
            marker = None if (st is None or st.marker == "none") else st.marker
            ms = st.markersize if st else 3.5
            alpha = st.alpha if st else 1.0
            zorder = st.zorder if (st and st.zorder is not None) else (3 + idx)

            if plot_kind == "hist":
                ax.hist(
                    s_t.y[np.isfinite(s_t.y)],
                    bins=panel.marginal_bins,
                    color=col,
                    alpha=min(alpha, 0.55),
                    density=True,
                    edgecolor=col,
                    linewidth=0.7,
                    label=s_t.name,
                )
                continue

            if plot_kind == "bar":
                ax.bar(s_t.x, s_t.y, color=col, alpha=min(alpha, 0.85), label=s_t.name, width=0.8)
                continue

            if plot_kind == "scatter":
                ax.scatter(
                    s_t.x,
                    s_t.y,
                    color=col,
                    s=(ms * 2.5) ** 2 / 2.0,
                    alpha=alpha,
                    marker=marker or "o",
                    label=s_t.name,
                    edgecolors="none",
                    zorder=zorder,
                )
                all_y_for_marginal.append((s_t.y, col))
                continue

            # Line plot (with optional smoothing & error bars/bands)
            if panel.smoothing_method != "none":
                y_smooth = smooth_series(s_t.y, panel.smoothing_method, panel.smoothing_window)
                if panel.show_raw_under_smooth:
                    ax.plot(
                        s_t.x,
                        s_t.y,
                        color=col,
                        linewidth=max(0.5, lw * 0.6),
                        alpha=panel.raw_alpha,
                        zorder=zorder - 1,
                    )
                ax.plot(
                    s_t.x,
                    y_smooth,
                    color=col,
                    linewidth=lw * 1.1,
                    linestyle=ls,
                    marker=marker,
                    markersize=ms,
                    alpha=alpha,
                    label=s_t.name,
                    zorder=zorder,
                )
                y_for_stats = y_smooth
            else:
                if s_t.dy is not None and panel.show_errorbars:
                    if panel.errorbar_style == "band":
                        ax.fill_between(
                            s_t.x,
                            s_t.y - s_t.dy,
                            s_t.y + s_t.dy,
                            color=col,
                            alpha=0.22,
                            linewidth=0,
                            zorder=zorder - 1,
                        )
                        ax.plot(
                            s_t.x,
                            s_t.y,
                            color=col,
                            linewidth=lw,
                            linestyle=ls,
                            marker=marker,
                            markersize=ms,
                            alpha=alpha,
                            label=s_t.name,
                            zorder=zorder,
                        )
                    else:
                        ax.errorbar(
                            s_t.x,
                            s_t.y,
                            yerr=s_t.dy,
                            xerr=s_t.dx,
                            color=col,
                            linewidth=lw,
                            linestyle=ls,
                            marker=marker,
                            markersize=ms,
                            elinewidth=lw * 0.7,
                            capsize=2.0,
                            alpha=alpha,
                            label=s_t.name,
                            zorder=zorder,
                        )
                else:
                    ax.plot(
                        s_t.x,
                        s_t.y,
                        color=col,
                        linewidth=lw,
                        linestyle=ls,
                        marker=marker,
                        markersize=ms,
                        alpha=alpha,
                        label=s_t.name,
                        zorder=zorder,
                    )
                y_for_stats = s_t.y

            if st and st.fill_under:
                ax.fill_between(s_t.x, 0, y_for_stats, color=col, alpha=0.14, zorder=1)

            if panel.show_mean_line:
                start = int(len(s_t.y) * panel.equilibration_fraction)
                mean_val = float(np.nanmean(s_t.y[start:]))
                ax.axhline(mean_val, color=col, linestyle="--", linewidth=0.75, alpha=0.75)

            all_y_for_marginal.append((s_t.y, col))

    # Scales & limits
    ax.set_xscale(panel.x_scale)
    ax.set_yscale(panel.y_scale)
    if panel.x_min is not None or panel.x_max is not None:
        ax.set_xlim(panel.x_min, panel.x_max)
    if panel.y_min is not None or panel.y_max is not None:
        ax.set_ylim(panel.y_min, panel.y_max)

    # Axis labels (auto unit update when user didn't override)
    x_lbl = (
        panel.x_label
        if panel.x_label is not None
        else update_label_unit(first_ds.x_label, first_ds.x_unit, panel.x_unit)
    )
    y_lbl = (
        panel.y_label
        if panel.y_label is not None
        else update_label_unit(first_ds.y_label, first_ds.y_unit, panel.y_unit)
    )
    if panel.plot_type == "hist":
        ax.set_xlabel(panel.x_label if panel.x_label is not None else y_lbl)
        ax.set_ylabel(panel.y_label if panel.y_label is not None else "Probability Density")
    else:
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)

    if panel.title:
        ax.set_title(panel.title, pad=5)

    if panel.grid:
        ax.grid(True, which="major")
    if panel.minor_ticks and panel.x_scale == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if panel.minor_ticks and panel.y_scale == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    _apply_annotations(ax, panel.annotations)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if panel.show_legend and handles:
        leg = ax.legend(
            loc=panel.legend_loc,
            ncol=panel.legend_ncol,
            frameon=panel.legend_frame,
            edgecolor="#CBD5E1",
            facecolor="white",
            framealpha=0.9,
        )
        if leg and leg.get_frame():
            leg.get_frame().set_linewidth(0.6)

    # Marginal distribution histogram / KDE on the right side (MD publication standard)
    if panel.marginal_hist and all_y_for_marginal and panel.plot_type != "hist":
        divider = make_axes_locatable(ax)
        ax_marg = divider.append_axes("right", size="18%", pad=0.06, sharey=ax)
        y_lo, y_hi = ax.get_ylim()
        y_grid = np.linspace(y_lo, y_hi, 200)

        for y_arr, col in all_y_for_marginal:
            start = int(len(y_arr) * panel.equilibration_fraction)
            vals = y_arr[start:]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            ax_marg.hist(
                vals,
                bins=panel.marginal_bins,
                range=(y_lo, y_hi),
                orientation="horizontal",
                density=True,
                color=col,
                alpha=0.28,
                edgecolor="none",
            )
            if panel.marginal_kde and np.nanstd(vals) > 1e-12:
                try:
                    kde = gaussian_kde(vals)
                    ax_marg.plot(kde(y_grid), y_grid, color=col, linewidth=max(0.8, default_lw * 0.85))
                except Exception:
                    pass

        ax_marg.tick_params(axis="y", labelleft=False, left=False, right=False)
        ax_marg.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
        for sp in ("top", "right", "bottom"):
            ax_marg.spines[sp].set_visible(False)
        ax_marg.spines["left"].set_linewidth(plt.rcParams["axes.linewidth"] * 0.8)
        ax_marg.set_xlabel("Dist.", fontsize=max(6.0, plt.rcParams["xtick.labelsize"] * 0.9))


def render_to_figure(spec: FigureSpec, datasets: dict[str, ParsedDataset]) -> Figure:
    """Create a publication-quality Matplotlib Figure from a FigureSpec and parsed datasets."""
    preset = get_journal_preset(spec.journal)
    palette = get_palette(spec.palette)
    rc = build_rc_params(
        preset,
        font_family=spec.font_family,
        font_size=spec.font_size,
        line_width=spec.line_width,
    )
    default_lw = rc["lines.linewidth"]
    figsize = _compute_figure_size_inches(spec)

    with matplotlib.rc_context(rc):
        fig, axes_raw = plt.subplots(
            nrows=spec.nrows,
            ncols=spec.ncols,
            figsize=figsize,
            sharex=spec.share_x,
            sharey=spec.share_y,
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = list(axes_raw.ravel())

        total_slots = spec.nrows * spec.ncols
        multi_panel = total_slots > 1

        for idx, ax in enumerate(axes_flat):
            if idx >= len(spec.panels):
                ax.set_visible(False)
                continue

            panel = spec.panels[idx]
            xy_items, matrix_ds = _resolve_panel_series(panel, datasets)

            if matrix_ds is not None and panel.plot_type in {"auto", "matrix", "contour"}:
                _render_matrix_panel(ax, panel, matrix_ds, preset.axes_linewidth)
            else:
                _render_xy_panel(ax, panel, xy_items, palette, default_lw)

            # Panel labels (a, b, c, ...)
            if multi_panel and spec.panel_labels != "none":
                lbl = _format_panel_label(idx, spec.panel_labels)
                if lbl:
                    ax.text(
                        -0.14,
                        1.06,
                        lbl,
                        transform=ax.transAxes,
                        fontsize=preset.panel_label_size,
                        fontweight="bold",
                        va="bottom",
                        ha="left",
                    )

        if spec.suptitle:
            fig.suptitle(spec.suptitle, fontsize=rc["axes.titlesize"] * 1.1, fontweight="bold")

        return fig


def render_figure_bytes(
    spec: FigureSpec,
    datasets: dict[str, ParsedDataset],
    *,
    fmt: str = "svg",
    dpi: int | None = None,
) -> bytes:
    """Render FigureSpec directly to vector (SVG/PDF/EPS) or high-DPI raster (PNG/TIFF) bytes."""
    fmt_clean = fmt.lower().lstrip(".")
    if fmt_clean == "tif":
        fmt_clean = "tiff"
    if fmt_clean not in {"svg", "pdf", "eps", "png", "tiff"}:
        raise ValueError(f"Unsupported export format: {fmt}")

    out_dpi = dpi or spec.dpi
    with _RENDER_LOCK:
        fig = render_to_figure(spec, datasets)
        try:
            buf = io.BytesIO()
            if fmt_clean == "tiff":
                # Render crisp high-DPI PNG first then save LZW-compressed TIFF for journals
                png_buf = io.BytesIO()
                fig.savefig(
                    png_buf,
                    format="png",
                    dpi=out_dpi,
                    transparent=spec.transparent,
                    bbox_inches="tight",
                    pad_inches=0.04,
                )
                png_buf.seek(0)
                img = Image.open(png_buf)
                if not spec.transparent:
                    img = img.convert("RGB")
                img.save(buf, format="TIFF", compression="tiff_lzw", dpi=(out_dpi, out_dpi))
            else:
                fig.savefig(
                    buf,
                    format=fmt_clean,
                    dpi=out_dpi,
                    transparent=spec.transparent,
                    bbox_inches="tight",
                    pad_inches=0.04,
                )
            return buf.getvalue()
        finally:
            plt.close(fig)


def save_figure(
    spec: FigureSpec,
    datasets: dict[str, ParsedDataset],
    output: str | Path,
    *,
    fmt: str | None = None,
    dpi: int | None = None,
) -> Path:
    """Render and write a publication figure to disk."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fmt = fmt or out_path.suffix.lstrip(".") or "pdf"
    data = render_figure_bytes(spec, datasets, fmt=resolved_fmt, dpi=dpi)
    out_path.write_bytes(data)
    return out_path


# ---------------------------------------------------------------------------
# High-level Python API helpers (import gmxgrace as gg; gg.plot(...))
# ---------------------------------------------------------------------------


def quick_plot(
    files: str | Path | list[str | Path],
    *,
    output: str | Path | None = None,
    journal: str = "nature",
    column_width: str = "single",
    palette: str = "okabe-ito",
    nxy: bool = False,
    x_unit: str = "original",
    y_unit: str = "original",
    smoothing: str = "none",
    window: int = 25,
    aggregate: str = "none",
    marginal: bool = False,
    title: str = "",
    x_label: str | None = None,
    y_label: str | None = None,
    dpi: int = 300,
    edr_columns: list[str] | None = None,
    **panel_kwargs: Any,
) -> Figure | Path:
    """One-liner Python API to plot one or more GROMACS files with publication styling."""
    from gmxgrace.parsers import parse_file

    file_list = [files] if isinstance(files, (str, Path)) else list(files)
    datasets: dict[str, ParsedDataset] = {}
    ds_ids: list[str] = []
    for i, fp in enumerate(file_list):
        ds = parse_file(fp, nxy_hint=nxy, edr_columns=edr_columns)
        ds_id = f"ds_{i}"
        datasets[ds_id] = ds
        ds_ids.append(ds_id)

    panel = PanelSpec(
        id="panel-1",
        title=title,
        dataset_ids=ds_ids,
        x_label=x_label,
        y_label=y_label,
        x_unit=x_unit,  # type: ignore[arg-type]
        y_unit=y_unit,  # type: ignore[arg-type]
        smoothing_method=smoothing,  # type: ignore[arg-type]
        smoothing_window=window,
        aggregate=aggregate,  # type: ignore[arg-type]
        marginal_hist=marginal,
        **panel_kwargs,
    )
    spec = FigureSpec(
        journal=journal,  # type: ignore[arg-type]
        column_width=column_width,  # type: ignore[arg-type]
        palette=palette,
        dpi=dpi,
        panels=[panel],
    )
    if output is not None:
        return save_figure(spec, datasets, output)
    return render_to_figure(spec, datasets)
