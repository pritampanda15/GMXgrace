"""Unified publication-quality Matplotlib rendering engine for CLI, Python API, and Web Studio."""

from __future__ import annotations

import io
import json
import threading
from html import escape as html_escape
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

from gmxgrace.analysis import (
    aggregate_replicates,
    compute_convergence,
    compute_fel_from_series,
    compute_pairwise_tests,
    minmax_downsample,
    minmax_downsample_band,
    smooth_series,
    unit_scale,
    update_label_unit,
)
from gmxgrace.models import (
    AnnotationSpec,
    DomainTrackSpec,
    FigureSpec,
    PanelSpec,
    ParsedDataset,
    SeriesData,
    SeriesStyleSpec,
    ZoomInsetSpec,
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
    max_plot_points: int = 12000,
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

    # Stride very large trajectories for fast 2D KDE & scatter rendering
    kde_step = max(1, len(xv) // 25000)
    xk, yk = xv[::kde_step], yv[::kde_step]
    try:
        kde = gaussian_kde(np.vstack([xk, yk]))
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
    sc_step = max(1, len(xv) // max(2000, max_plot_points)) if max_plot_points > 0 else 1
    ax.scatter(xv[::sc_step], yv[::sc_step], s=3.5, color="#111827", alpha=0.18, edgecolors="none", zorder=4)
    if is_rama:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
        ax.axhline(0, color="#64748B", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axvline(0, color="#64748B", linewidth=0.5, linestyle=":", alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


def _render_fel2d(
    ax: Axes,
    panel: PanelSpec,
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]],
    xy_items: list[tuple[SeriesData, SeriesStyleSpec | None, ParsedDataset]],
) -> tuple[str, str]:
    """Compute and plot a 2D Gibbs Free Energy Landscape from two .xvg collective-variable series."""
    if len(transformed_series) < 2:
        ax.text(
            0.5,
            0.5,
            "FEL from 2 XVGs requires at least 2 series\n(e.g. rmsd.xvg + gyrate.xvg or pc1 + pc2)",
            ha="center",
            va="center",
            color="#64748B",
            fontsize=8.0,
            transform=ax.transAxes,
        )
        return ("CV 1", "CV 2")

    s_x = transformed_series[0][0]
    s_y = transformed_series[1][0]
    ds_x = xy_items[0][2]
    ds_y = xy_items[1][2]
    time_unit = panel.x_unit if panel.x_unit != "original" else ds_x.x_unit

    fel = compute_fel_from_series(
        s_x,
        s_y,
        temp_k=panel.fel_temp_k,
        bins=panel.fel_bins,
        unit=panel.fel_unit,
        eq_fraction=panel.equilibration_fraction,
        time_unit=time_unit,
    )
    xx, yy = np.meshgrid(fel.x_centers, fel.y_centers)
    vmin = panel.v_min if panel.v_min is not None else 0.0
    vmax = panel.v_max if panel.v_max is not None else float(np.nanmax(fel.dg_grid))

    cf = ax.contourf(xx, yy, fel.dg_grid, levels=panel.contour_levels, cmap=panel.cmap, vmin=vmin, vmax=vmax)
    if panel.show_contours:
        ax.contour(
            xx,
            yy,
            fel.dg_grid,
            levels=panel.contour_levels,
            colors="#111827",
            linewidths=0.45,
            alpha=0.5,
        )

    if panel.fel_mark_minima and fel.minima:
        for m in fel.minima:
            ax.scatter(
                [m.x],
                [m.y],
                marker="*" if m.rank == 1 else "o",
                s=75 if m.rank == 1 else 36,
                color="#FFFFFF",
                edgecolors="#0F172A",
                linewidths=0.9,
                zorder=8,
            )
            ax.annotate(
                f"M{m.rank}",
                (m.x, m.y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=max(6.5, plt.rcParams["legend.fontsize"]),
                fontweight="bold",
                color="#0F172A",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.85, "lw": 0.5},
                zorder=9,
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = ax.figure.colorbar(cf, cax=cax)
    unit_disp = "kcal mol⁻¹" if panel.fel_unit == "kcal/mol" else "kJ mol⁻¹"
    cbar.set_label(panel.colorbar_label or f"ΔG ({unit_disp})")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)

    xl = update_label_unit(ds_x.y_label, ds_x.y_unit, panel.y_unit) or s_x.name
    yl = update_label_unit(ds_y.y_label, ds_y.y_unit, panel.y_unit) or s_y.name
    return (xl, yl)


def _render_comparison(
    ax: Axes,
    panel: PanelSpec,
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]],
    mode: str,
) -> None:
    """Render publication Violin (raincloud) or Box comparison across equilibrated MD systems + p-values."""
    groups: list[tuple[str, np.ndarray, str]] = []
    for s_t, _, col in transformed_series:
        start = int(len(s_t.y) * panel.equilibration_fraction)
        vals = s_t.y[start:]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            groups.append((s_t.name, vals, col))

    if not groups:
        return

    positions = np.arange(1, len(groups) + 1, dtype=float)
    data_list = [g[1] for g in groups]
    labels = [g[0] for g in groups]
    colors = [g[2] for g in groups]

    rng = np.random.default_rng(42)

    if mode == "violin":
        vp = ax.violinplot(data_list, positions=positions, widths=0.68, showmeans=False, showextrema=False, showmedians=False)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(0.32)
            body.set_linewidth(0.9)
        for pos, vals, col in zip(positions, data_list, colors):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos, q1, q3, color=col, linewidth=3.4, zorder=4)
            ax.scatter([pos], [med], color="white", edgecolors=col, s=22, linewidths=1.1, zorder=5)
            # Subtle subsampled raincloud jitter
            sub = vals[:: max(1, len(vals) // 80)]
            jitter = rng.normal(0.0, 0.06, size=len(sub))
            ax.scatter(pos + jitter, sub, color=col, s=5.0, alpha=0.22, edgecolors="none", zorder=2)
    else:
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0F172A", "linewidth": 1.3},
            whiskerprops={"color": "#475569", "linewidth": 0.9},
            capprops={"color": "#475569", "linewidth": 0.9},
        )
        for box, col in zip(bp["boxes"], colors):
            box.set_facecolor(col)
            box.set_alpha(0.38)
            box.set_edgecolor(col)
            box.set_linewidth(1.0)
        for pos, vals, col in zip(positions, data_list, colors):
            sub = vals[:: max(1, len(vals) // 60)]
            jitter = rng.normal(0.0, 0.055, size=len(sub))
            ax.scatter(pos + jitter, sub, color=col, s=5.0, alpha=0.25, edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15 if len(labels) > 3 else 0)
    ax.set_xlim(0.4, len(groups) + 0.6)

    # Automatic pairwise significance brackets
    if panel.show_pvalues and len(groups) >= 2:
        tests = compute_pairwise_tests([(g[0], g[1]) for g in groups], method=panel.stat_test)
        if tests:
            y_lo, y_hi = ax.get_ylim()
            span = max(y_hi - y_lo, 1e-6)
            step = span * 0.11
            cur_y = y_hi + span * 0.04
            for res in tests:
                x1 = positions[res.group_a]
                x2 = positions[res.group_b]
                h = span * 0.025
                ax.plot([x1, x1, x2, x2], [cur_y, cur_y + h, cur_y + h, cur_y], color="#334155", linewidth=0.8)
                p_txt = f"{res.stars}" if res.stars != "ns" else "ns"
                ax.text(
                    0.5 * (x1 + x2),
                    cur_y + h + span * 0.008,
                    p_txt,
                    ha="center",
                    va="bottom",
                    fontsize=max(6.5, plt.rcParams["legend.fontsize"]),
                    fontweight="semibold" if res.stars != "ns" else "normal",
                    color="#0F172A" if res.stars != "ns" else "#64748B",
                )
                cur_y += step
            ax.set_ylim(y_lo, cur_y + span * 0.04)


def _render_delta(
    ax: Axes,
    panel: PanelSpec,
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]],
    default_lw: float,
) -> None:
    """Render Δ-difference curve (Series 2 − Series 1, e.g. ΔRMSF Mutant − WT) with red/blue fills."""
    if len(transformed_series) < 2:
        ax.text(
            0.5,
            0.5,
            "Δ-Difference mode requires at least 2 series\n(e.g. Wild-Type and Mutant RMSF)",
            ha="center",
            va="center",
            color="#64748B",
            fontsize=8.0,
            transform=ax.transAxes,
        )
        return

    s0 = transformed_series[0][0]
    s1 = transformed_series[1][0]
    if len(s0.x) == len(s1.x) and np.allclose(s0.x, s1.x, rtol=1e-4, atol=1e-6):
        x = s0.x
        dy = s1.y - s0.y
    else:
        x = s0.x
        order = np.argsort(s1.x)
        dy = np.interp(x, s1.x[order], s1.y[order]) - s0.y

    if panel.smoothing_method != "none":
        dy = smooth_series(dy, panel.smoothing_method, panel.smoothing_window)

    ax.axhline(0.0, color="#334155", linewidth=0.8, linestyle="-", zorder=2)
    ax.fill_between(x, 0.0, dy, where=(dy >= 0), interpolate=True, color="#DC2626", alpha=0.30, label=f"↑ {s1.name} > {s0.name}")
    ax.fill_between(x, 0.0, dy, where=(dy <= 0), interpolate=True, color="#1D4ED8", alpha=0.30, label=f"↓ {s1.name} < {s0.name}")
    ax.plot(x, dy, color="#0F172A", linewidth=max(0.8, default_lw * 0.85), zorder=4)


def _render_convergence(
    ax: Axes,
    panel: PanelSpec,
    transformed_series: list[tuple[SeriesData, SeriesStyleSpec | None, str]],
    default_lw: float,
    max_plot_points: int,
) -> None:
    """Render cumulative running mean + block averages (± SD) and convergence status badge."""
    status_lines: list[str] = []
    for s_t, _, col in transformed_series:
        start = int(len(s_t.y) * panel.equilibration_fraction)
        xv = s_t.x[start:]
        yv = s_t.y[start:]
        valid = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[valid], yv[valid]
        if len(yv) < 6:
            continue
        cum_mean = np.cumsum(yv) / np.arange(1, len(yv) + 1)
        xr, yr = minmax_downsample(xv, yv, max_plot_points)
        xc, yc = minmax_downsample(xv, cum_mean, max_plot_points)
        ax.plot(xr, yr, color=col, alpha=0.18, linewidth=max(0.45, default_lw * 0.5), zorder=1)
        ax.plot(xc, yc, color=col, linewidth=default_lw * 1.15, label=f"{s_t.name} (cum. mean)", zorder=3)

        conv = compute_convergence(s_t.x, s_t.y, eq_fraction=panel.equilibration_fraction, n_blocks=panel.convergence_blocks)
        ax.errorbar(
            conv.block_centers,
            conv.block_means,
            yerr=conv.block_stds,
            fmt="o-",
            color=col,
            markersize=3.8,
            linewidth=0.85,
            capsize=2.2,
            alpha=0.92,
            zorder=5,
        )
        status_lines.append(f"{s_t.name}: {conv.status} (SEM={conv.block_sem:.3g}, N_eff={conv.n_eff:.0f})")

    if status_lines:
        ax.text(
            0.02,
            0.03,
            "\n".join(status_lines[:3]),
            transform=ax.transAxes,
            fontsize=max(5.8, plt.rcParams["legend.fontsize"] * 0.85),
            va="bottom",
            ha="left",
            color="#0F172A",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9, "lw": 0.6},
            zorder=8,
        )


def _add_domain_tracks(ax: Axes, tracks: list[DomainTrackSpec]) -> None:
    """Attach a colored protein domain / motif track strip along the top of the panel."""
    if not tracks:
        return
    divider = make_axes_locatable(ax)
    ax_dom = divider.append_axes("top", size="11%", pad=0.05, sharex=ax)
    ax_dom.set_ylim(0.0, 1.0)
    x0, x1 = ax.get_xlim()
    x_span = max(abs(x1 - x0), 1e-6)
    base_fs = max(5.5, plt.rcParams["xtick.labelsize"] * 0.85)
    for tr in tracks:
        width = max(tr.end - tr.start, 0.5)
        frac = width / x_span
        ax_dom.barh(0.5, width, left=tr.start, height=0.76, color=tr.color, alpha=0.88, edgecolor="white", linewidth=0.6)
        # Scale label font cleanly for narrow domains/loops so text never clips
        char_budget = max(3, int(frac * 58))
        fs = base_fs if len(tr.label) <= char_budget else max(4.6, base_fs * (char_budget / len(tr.label)))
        ax_dom.text(
            tr.start + 0.5 * width,
            0.5,
            tr.label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="semibold",
            color="white",
            clip_on=True,
        )
    ax_dom.set_yticks([])
    ax_dom.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)
    for sp in ("top", "right", "left", "bottom"):
        ax_dom.spines[sp].set_visible(False)


def _add_zoom_inset(
    ax: Axes,
    inset: ZoomInsetSpec,
    plotted_curves: list[tuple[np.ndarray, np.ndarray, str, float, str]],
) -> None:
    """Add a magnified zoom inset box with publication connector lines."""
    if not inset.enabled or not plotted_curves:
        return
    if inset.x_min is None or inset.x_max is None:
        return

    w = inset.width_pct / 100.0
    h = inset.height_pct / 100.0
    loc_bounds = {
        "upper right": [1.0 - w - 0.04, 1.0 - h - 0.05, w, h],
        "upper left": [0.05, 1.0 - h - 0.05, w, h],
        "lower right": [1.0 - w - 0.04, 0.07, w, h],
        "lower left": [0.05, 0.07, w, h],
    }
    bounds = loc_bounds.get(inset.loc, loc_bounds["upper right"])
    axins = ax.inset_axes(bounds)
    for xp, yp, col, lw, ls in plotted_curves:
        axins.plot(xp, yp, color=col, linewidth=max(0.7, lw * 0.9), linestyle=ls)

    axins.set_xlim(inset.x_min, inset.x_max)
    if inset.y_min is not None or inset.y_max is not None:
        axins.set_ylim(inset.y_min, inset.y_max)
    else:
        # Auto-fit Y to data inside the zoomed X window
        ys_in: list[np.ndarray] = []
        for xp, yp, *_ in plotted_curves:
            m = (xp >= inset.x_min) & (xp <= inset.x_max) & np.isfinite(yp)
            if np.any(m):
                ys_in.append(yp[m])
        if ys_in:
            cat = np.concatenate(ys_in)
            pad = max(float(np.max(cat) - np.min(cat)) * 0.12, 1e-4)
            axins.set_ylim(float(np.min(cat)) - pad, float(np.max(cat)) + pad)

    axins.tick_params(labelsize=max(5.2, plt.rcParams["xtick.labelsize"] * 0.78), pad=1.5)
    axins.set_facecolor("white")
    for sp in ("top", "right", "bottom", "left"):
        axins.spines[sp].set_visible(True)
        axins.spines[sp].set_edgecolor("#475569")
        axins.spines[sp].set_linewidth(0.7)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="#475569", linewidth=0.75, alpha=0.8)
    except Exception:
        pass


def _render_xy_panel(
    ax: Axes,
    panel: PanelSpec,
    xy_items: list[tuple[SeriesData, SeriesStyleSpec | None, ParsedDataset]],
    palette: list[str],
    default_lw: float,
    max_plot_points: int = 12000,
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
        is_right = bool(st and st.y_axis == "right")
        ys = unit_scale(ds.y_unit, panel.y2_unit if is_right else panel.y_unit)
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
    plotted_curves_for_inset: list[tuple[np.ndarray, np.ndarray, str, float, str]] = []
    ax2: Axes | None = None
    custom_x_lbl: str | None = None
    custom_y_lbl: str | None = None

    # Special biophysics plot modes
    if panel.plot_type == "fel2d":
        custom_x_lbl, custom_y_lbl = _render_fel2d(ax, panel, transformed_series, xy_items)
    elif panel.plot_type in {"violin", "box"}:
        _render_comparison(ax, panel, transformed_series, mode=panel.plot_type)
        custom_x_lbl = ""
    elif panel.plot_type == "delta":
        _render_delta(ax, panel, transformed_series, default_lw)
        base_y = update_label_unit(first_ds.y_label, first_ds.y_unit, panel.y_unit)
        custom_y_lbl = f"Δ {base_y}"
    elif panel.plot_type == "convergence":
        _render_convergence(ax, panel, transformed_series, default_lw, max_plot_points)
    elif panel.aggregate != "none" and len(transformed_series) >= 2:
        if panel.aggregate_show_individual:
            for s_t, st, col in transformed_series:
                xp, yp = minmax_downsample(s_t.x, s_t.y, max_plot_points)
                ax.plot(
                    xp,
                    yp,
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

            xb, cb, lob, hib = minmax_downsample_band(agg.x, center, lower, upper, max_plot_points)
            ax.fill_between(xb, lob, hib, color=agg_col, alpha=0.22, linewidth=0, zorder=3)
            xc, yc = minmax_downsample(agg.x, center, max_plot_points)
            ax.plot(xc, yc, color=agg_col, linewidth=default_lw * 1.15, label=agg_lbl, zorder=4)
            plotted_curves_for_inset.append((agg.x, center, agg_col, default_lw * 1.15, "-"))
            all_y_for_marginal.append((center, agg_col))

            if panel.show_mean_line:
                start = int(len(center) * panel.equilibration_fraction)
                mean_val = float(np.nanmean(center[start:]))
                ax.axhline(mean_val, color=agg_col, linestyle="--", linewidth=0.8, alpha=0.8)
    else:
        # Standard multi-series plotting (supports Twin-Y right axis per series!)
        plot_kind = panel.plot_type
        if plot_kind == "auto":
            plot_kind = "density2d" if first_ds.detected_type == "ramachandran" else "line"

        has_right = any(bool(st and st.y_axis == "right") for _, st, _ in transformed_series)
        first_right_color: str | None = None
        first_right_ds: ParsedDataset | None = None
        if has_right and plot_kind not in {"density2d", "hist"}:
            ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax2.spines["top"].set_visible(ax.spines["top"].get_visible())

        if plot_kind == "density2d":
            _render_density2d(ax, panel, transformed_series, first_ds, max_plot_points=max_plot_points)
        for idx, (s_t, st, col) in enumerate([] if plot_kind == "density2d" else transformed_series):
            is_right = bool(ax2 is not None and st and st.y_axis == "right")
            target_ax = ax2 if is_right else ax
            if is_right and first_right_color is None:
                first_right_color = col
                first_right_ds = xy_items[idx][2]

            lw = (st.linewidth if st and st.linewidth is not None else default_lw)
            ls = st.linestyle if st else "-"
            marker = None if (st is None or st.marker == "none") else st.marker
            ms = st.markersize if st else 3.5
            alpha = st.alpha if st else 1.0
            zorder = st.zorder if (st and st.zorder is not None) else (3 + idx)

            if plot_kind == "hist":
                target_ax.hist(
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
                target_ax.bar(s_t.x, s_t.y, color=col, alpha=min(alpha, 0.85), label=s_t.name, width=0.8)
                continue

            if plot_kind == "scatter":
                sc_step = max(1, len(s_t.x) // max(2000, max_plot_points)) if max_plot_points > 0 else 1
                target_ax.scatter(
                    s_t.x[::sc_step],
                    s_t.y[::sc_step],
                    color=col,
                    s=(ms * 2.5) ** 2 / 2.0,
                    alpha=alpha,
                    marker=marker or "o",
                    label=s_t.name,
                    edgecolors="none",
                    zorder=zorder,
                )
                if not is_right:
                    all_y_for_marginal.append((s_t.y, col))
                continue

            # Line plot (with optional smoothing & error bars/bands)
            if panel.smoothing_method != "none":
                y_smooth = smooth_series(s_t.y, panel.smoothing_method, panel.smoothing_window)
                if panel.show_raw_under_smooth:
                    xr, yr = minmax_downsample(s_t.x, s_t.y, max_plot_points)
                    target_ax.plot(
                        xr,
                        yr,
                        color=col,
                        linewidth=max(0.5, lw * 0.6),
                        alpha=panel.raw_alpha,
                        zorder=zorder - 1,
                    )
                xs_p, ys_p = minmax_downsample(s_t.x, y_smooth, max_plot_points)
                target_ax.plot(
                    xs_p,
                    ys_p,
                    color=col,
                    linewidth=lw * 1.1,
                    linestyle=ls,
                    marker=marker,
                    markersize=ms,
                    alpha=alpha,
                    label=s_t.name,
                    zorder=zorder,
                )
                if not is_right:
                    plotted_curves_for_inset.append((s_t.x, y_smooth, col, lw * 1.1, ls))
                y_for_stats = y_smooth
            else:
                if s_t.dy is not None and panel.show_errorbars:
                    if panel.errorbar_style == "band":
                        xb, cb, lob, hib = minmax_downsample_band(
                            s_t.x, s_t.y, s_t.y - s_t.dy, s_t.y + s_t.dy, max_plot_points
                        )
                        target_ax.fill_between(
                            xb,
                            lob,
                            hib,
                            color=col,
                            alpha=0.22,
                            linewidth=0,
                            zorder=zorder - 1,
                        )
                        xp, yp = minmax_downsample(s_t.x, s_t.y, max_plot_points)
                        target_ax.plot(
                            xp,
                            yp,
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
                        target_ax.errorbar(
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
                    xp, yp = minmax_downsample(s_t.x, s_t.y, max_plot_points)
                    target_ax.plot(
                        xp,
                        yp,
                        color=col,
                        linewidth=lw,
                        linestyle=ls,
                        marker=marker,
                        markersize=ms,
                        alpha=alpha,
                        label=s_t.name,
                        zorder=zorder,
                    )
                if not is_right:
                    plotted_curves_for_inset.append((s_t.x, s_t.y, col, lw, ls))
                y_for_stats = s_t.y

            if st and st.fill_under:
                xf, yf = minmax_downsample(s_t.x, y_for_stats, max_plot_points)
                target_ax.fill_between(xf, 0, yf, color=col, alpha=0.14, zorder=1)

            if panel.show_mean_line:
                start = int(len(s_t.y) * panel.equilibration_fraction)
                mean_val = float(np.nanmean(s_t.y[start:]))
                target_ax.axhline(mean_val, color=col, linestyle="--", linewidth=0.75, alpha=0.75)

            if not is_right:
                all_y_for_marginal.append((s_t.y, col))

        if ax2 is not None:
            y2_lbl = panel.y2_label
            if not y2_lbl and first_right_ds is not None:
                y2_lbl = update_label_unit(first_right_ds.y_label, first_right_ds.y_unit, panel.y2_unit)
            ax2.set_ylabel(y2_lbl or "Right Y", color=first_right_color or "#111827")
            if first_right_color:
                ax2.tick_params(axis="y", colors=first_right_color)
                ax2.spines["right"].set_edgecolor(first_right_color)
            if panel.y2_min is not None or panel.y2_max is not None:
                ax2.set_ylim(panel.y2_min, panel.y2_max)
            if panel.minor_ticks:
                ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # Scales & limits
    if panel.plot_type not in {"violin", "box"}:
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
        else (custom_x_lbl if custom_x_lbl is not None else update_label_unit(first_ds.x_label, first_ds.x_unit, panel.x_unit))
    )
    y_lbl = (
        panel.y_label
        if panel.y_label is not None
        else (custom_y_lbl if custom_y_lbl is not None else update_label_unit(first_ds.y_label, first_ds.y_unit, panel.y_unit))
    )
    if panel.plot_type == "hist":
        ax.set_xlabel(panel.x_label if panel.x_label is not None else y_lbl)
        ax.set_ylabel(panel.y_label if panel.y_label is not None else "Probability Density")
    else:
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)

    if panel.title and not panel.domain_tracks:
        ax.set_title(panel.title, pad=5)

    if panel.grid:
        ax.grid(True, which="major")
    if panel.minor_ticks and panel.x_scale == "linear" and panel.plot_type not in {"violin", "box"}:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if panel.minor_ticks and panel.y_scale == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    _apply_annotations(ax, panel.annotations)

    # Legend (merges left & right Twin-Y handles when ax2 is present; avoids zoom inset corner)
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles = handles + h2
        labels = labels + l2
    if panel.show_legend and handles and panel.plot_type not in {"violin", "box", "fel2d"}:
        resolved_loc = panel.legend_loc
        if resolved_loc == "best" and panel.zoom_inset is not None and panel.zoom_inset.enabled:
            opposite_corner = {
                "lower right": "upper right",
                "upper right": "lower right",
                "lower left": "upper left",
                "upper left": "lower left",
            }
            resolved_loc = opposite_corner.get(panel.zoom_inset.loc, "upper right")
        leg = ax.legend(
            handles,
            labels,
            loc=resolved_loc,
            ncol=panel.legend_ncol,
            frameon=panel.legend_frame,
            edgecolor="#CBD5E1",
            facecolor="white",
            framealpha=0.9,
        )
        if leg and leg.get_frame():
            leg.get_frame().set_linewidth(0.6)

    # Optional magnified zoom inset
    if panel.zoom_inset is not None and panel.zoom_inset.enabled:
        _add_zoom_inset(ax, panel.zoom_inset, plotted_curves_for_inset)

    # Optional colored protein domain / secondary structure track along top
    if panel.domain_tracks:
        _add_domain_tracks(ax, panel.domain_tracks)
        if panel.title:
            ax.figure.axes[-1].set_title(panel.title, pad=4)

    # Marginal distribution histogram / KDE on the right side (MD publication standard)
    if panel.marginal_hist and all_y_for_marginal and ax2 is None and panel.plot_type not in {"hist", "violin", "box", "fel2d", "density2d"}:
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
                    kde_step = max(1, len(vals) // 50000)
                    kde = gaussian_kde(vals[::kde_step])
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

            ax.set_gid(f"gmx_ax_{idx}")
            panel = spec.panels[idx]
            xy_items, matrix_ds = _resolve_panel_series(panel, datasets)

            if matrix_ds is not None and panel.plot_type in {"auto", "matrix", "contour"}:
                _render_matrix_panel(ax, panel, matrix_ds, preset.axes_linewidth)
            else:
                _render_xy_panel(ax, panel, xy_items, palette, default_lw, max_plot_points=spec.max_plot_points)

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
            raw = buf.getvalue()
            if fmt_clean == "svg":
                axes_meta = []
                for ax in fig.axes:
                    gid = ax.get_gid()
                    if not gid or not gid.startswith("gmx_ax_"):
                        continue
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    axes_meta.append(
                        {
                            "gid": gid,
                            "xlim": [float(x0), float(x1)],
                            "ylim": [float(y0), float(y1)],
                            "xscale": ax.get_xscale(),
                            "yscale": ax.get_yscale(),
                            "xlabel": ax.get_xlabel(),
                            "ylabel": ax.get_ylabel(),
                            "title": ax.get_title(),
                        }
                    )
                if axes_meta:
                    meta_tag = f'<desc id="gmxgrace-axes-meta">{html_escape(json.dumps(axes_meta))}</desc>'
                    svg_str = raw.decode("utf-8", errors="replace")
                    idx_end = svg_str.rfind("</svg>")
                    if idx_end != -1:
                        svg_str = svg_str[:idx_end] + meta_tag + "\n</svg>"
                        raw = svg_str.encode("utf-8")
            return raw
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
