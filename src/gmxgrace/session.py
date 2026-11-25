"""Shared stateful studio session used identically by both FastAPI and Pyodide (GitHub Pages)."""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any

from gmxgrace.analysis import compute_convergence, compute_fel_from_series, unit_scale
from gmxgrace.codegen import generate_cli_command, generate_python_script, generate_recipe_dict
from gmxgrace.models import (
    AnnotationSpec,
    DomainTrackSpec,
    FigureSpec,
    PanelSpec,
    ParsedDataset,
    SeriesData,
    ZoomInsetSpec,
)
from gmxgrace.parsers import parse_bytes
from gmxgrace.plotting import render_figure_bytes
from gmxgrace.samples import SAMPLE_BUNDLES, get_sample_files
from gmxgrace.styles import list_presets_metadata

MAX_STORED_DATASETS = 250


class DatasetStore:
    """Bounded LRU in-memory store for uploaded & demo datasets."""

    def __init__(self, max_items: int = MAX_STORED_DATASETS) -> None:
        self._items: OrderedDict[str, ParsedDataset] = OrderedDict()
        self._max_items = max_items

    def put(self, dataset: ParsedDataset, dataset_id: str | None = None) -> str:
        ds_id = dataset_id or f"ds_{uuid.uuid4().hex[:8]}"
        self._items[ds_id] = dataset
        self._items.move_to_end(ds_id)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)
        return ds_id

    def get(self, dataset_id: str) -> ParsedDataset | None:
        ds = self._items.get(dataset_id)
        if ds is not None:
            self._items.move_to_end(dataset_id)
        return ds

    def delete(self, dataset_id: str) -> bool:
        return self._items.pop(dataset_id, None) is not None

    def all_map(self) -> dict[str, ParsedDataset]:
        return dict(self._items)

    def filenames_map(self) -> dict[str, str]:
        return {k: v.filename for k, v in self._items.items()}


class StudioSession:
    """Backend-agnostic controller for GMXgrace Studio (runs in FastAPI or Pyodide WASM)."""

    def __init__(self, max_items: int = MAX_STORED_DATASETS) -> None:
        self.store = DatasetStore(max_items=max_items)

    def get_presets(self) -> dict[str, Any]:
        meta = list_presets_metadata()
        meta["sample_bundles"] = SAMPLE_BUNDLES
        return meta

    def upload_files(self, files: list[tuple[str, bytes]], nxy: bool = False) -> dict[str, Any]:
        uploaded: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for fname, raw in files:
            try:
                ds = parse_bytes(raw, filename=fname or "unnamed.xvg", nxy_hint=nxy)
                ds_id = self.store.put(ds)
                summary = ds.to_summary_dict()
                summary["id"] = ds_id
                uploaded.append(summary)
            except Exception as exc:
                errors.append({"filename": fname, "error": str(exc)})
        return {"datasets": uploaded, "errors": errors}

    def delete_dataset(self, dataset_id: str) -> bool:
        return self.store.delete(dataset_id)

    def load_sample_bundle(self, bundle_id: str) -> dict[str, Any]:
        samples = get_sample_files()

        def _add(fname: str, nxy: bool = False) -> tuple[str, dict[str, Any]]:
            ds = parse_bytes(samples[fname].encode("utf-8"), filename=fname, nxy_hint=nxy)
            ds_id = self.store.put(ds)
            summary = ds.to_summary_dict()
            summary["id"] = ds_id
            return ds_id, summary

        loaded_summaries: list[dict[str, Any]] = []

        if bundle_id == "md-4panel":
            r1, s1 = _add("rmsd_rep1.xvg")
            r2, s2 = _add("rmsd_rep2.xvg")
            r3, s3 = _add("rmsd_rep3.xvg")
            rmsf_id, s_rmsf = _add("rmsf_wt_vs_mut.xvg")
            rg_id, s_rg = _add("gyrate.xvg")
            fel_id, s_fel = _add("gibbs_fel.xpm")
            loaded_summaries = [s1, s2, s3, s_rmsf, s_rg, s_fel]
            spec = FigureSpec(
                journal="nature",
                column_width="double",
                aspect_ratio=0.72,
                palette="nature-npg",
                nrows=2,
                ncols=2,
                panel_labels="lowercase",
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Backbone RMSD (n = 3)",
                        dataset_ids=[r1, r2, r3],
                        x_unit="ns",
                        y_unit="A",
                        aggregate="mean_sd",
                        aggregate_label="Mean ± SD (n=3)",
                        aggregate_show_individual=True,
                        marginal_hist=True,
                        equilibration_fraction=0.2,
                        annotations=[
                            AnnotationSpec(
                                kind="vspan", value=0.0, value_end=20.0, text="Eq.", color="#64748B", alpha=0.4
                            )
                        ],
                    ),
                    PanelSpec(
                        id="panel-2",
                        title="Per-Residue Flexibility",
                        dataset_ids=[rmsf_id],
                        y_unit="A",
                        annotations=[
                            AnnotationSpec(
                                kind="vline", value=46.0, text="G46A loop", color="#E64B35", linestyle="--"
                            )
                        ],
                    ),
                    PanelSpec(
                        id="panel-3",
                        title="Radius of Gyration",
                        dataset_ids=[rg_id],
                        x_unit="ns",
                        y_unit="A",
                        smoothing_method="moving_avg",
                        smoothing_window=21,
                        show_raw_under_smooth=True,
                        marginal_hist=True,
                        show_mean_line=True,
                    ),
                    PanelSpec(
                        id="panel-4",
                        title="Gibbs Free Energy Landscape",
                        dataset_ids=[fel_id],
                        plot_type="matrix",
                        cmap="RdYlBu_r",
                        show_contours=True,
                        contour_levels=9,
                        colorbar_label="ΔG (kJ mol⁻¹)",
                    ),
                ],
            )
        elif bundle_id == "rmsd-replicates":
            r1, s1 = _add("rmsd_rep1.xvg")
            r2, s2 = _add("rmsd_rep2.xvg")
            r3, s3 = _add("rmsd_rep3.xvg")
            loaded_summaries = [s1, s2, s3]
            spec = FigureSpec(
                journal="nature",
                column_width="one_half",
                aspect_ratio=0.65,
                palette="okabe-ito",
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Backbone RMSD Across Triplicate Trajectories",
                        dataset_ids=[r1, r2, r3],
                        x_unit="ns",
                        y_unit="A",
                        aggregate="mean_sd",
                        aggregate_label="Mean ± SD (n = 3)",
                        aggregate_show_individual=True,
                        marginal_hist=True,
                        equilibration_fraction=0.2,
                        show_mean_line=True,
                        annotations=[
                            AnnotationSpec(
                                kind="vspan",
                                value=0.0,
                                value_end=20.0,
                                text="Equilibration",
                                color="#64748B",
                                alpha=0.4,
                            )
                        ],
                        zoom_inset=ZoomInsetSpec(
                            enabled=True,
                            x_min=0.0,
                            x_max=20.0,
                            loc="lower right",
                            width_pct=36.0,
                            height_pct=34.0,
                        ),
                    )
                ],
            )
        elif bundle_id == "fel-xpm":
            fel_id, s_fel = _add("gibbs_fel.xpm")
            loaded_summaries = [s_fel]
            spec = FigureSpec(
                journal="acs",
                column_width="one_half",
                aspect_ratio=0.82,
                palette="acs-classic",
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Conformational Free Energy Landscape",
                        dataset_ids=[fel_id],
                        plot_type="matrix",
                        cmap="RdYlBu_r",
                        show_contours=True,
                        contour_levels=10,
                        colorbar_label="ΔG (kJ mol⁻¹)",
                    )
                ],
            )
        elif bundle_id == "dssp-xpm":
            dssp_id, s_dssp = _add("dssp.xpm")
            loaded_summaries = [s_dssp]
            spec = FigureSpec(
                journal="biophysj",
                column_width="one_half",
                aspect_ratio=0.62,
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Secondary Structure Evolution (DSSP)",
                        dataset_ids=[dssp_id],
                        plot_type="matrix",
                        show_contours=False,
                    )
                ],
            )
        elif bundle_id == "umbrella-pmf":
            pmf_id, s_pmf = _add("pmf_profile.xvg")
            hist_id, s_hist = _add("wham_histo.xvg", nxy=True)
            loaded_summaries = [s_pmf, s_hist]
            spec = FigureSpec(
                journal="acs",
                column_width="double",
                aspect_ratio=0.72,
                palette="acs-classic",
                nrows=1,
                ncols=2,
                panel_labels="parens_lower",
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Potential of Mean Force (WHAM)",
                        dataset_ids=[pmf_id],
                        y_unit="kcal/mol",
                        show_errorbars=True,
                        errorbar_style="band",
                        annotations=[AnnotationSpec(kind="hline", value=0.0, color="#64748B", linestyle=":")],
                    ),
                    PanelSpec(
                        id="panel-2",
                        title="Umbrella Sampling Windows",
                        dataset_ids=[hist_id],
                        show_legend=False,
                    ),
                ],
            )
        elif bundle_id == "rama-2d":
            rama_id, s_rama = _add("rama.xvg")
            loaded_summaries = [s_rama]
            spec = FigureSpec(
                journal="nature",
                column_width="one_half",
                aspect_ratio=0.92,
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Backbone Ramachandran Density",
                        dataset_ids=[rama_id],
                        plot_type="density2d",
                        x_label="φ (deg)",
                        y_label="ψ (deg)",
                        cmap="Blues",
                        show_contours=True,
                        contour_levels=8,
                        colorbar_label="Probability Density",
                        show_legend=False,
                    )
                ],
            )
        elif bundle_id == "fel-from-xvg":
            rmsd_id, s_rmsd = _add("rmsd_rep1.xvg")
            rg_id, s_rg = _add("gyrate.xvg")
            loaded_summaries = [s_rmsd, s_rg]
            spec = FigureSpec(
                journal="nature",
                column_width="one_half",
                aspect_ratio=0.82,
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="2D Free Energy Landscape (RMSD vs Rg)",
                        dataset_ids=[rmsd_id, rg_id],
                        plot_type="fel2d",
                        x_unit="ns",
                        y_unit="A",
                        fel_temp_k=300.0,
                        fel_bins=40,
                        fel_unit="kJ/mol",
                        fel_mark_minima=True,
                        cmap="RdYlBu_r",
                        show_contours=True,
                        contour_levels=10,
                    )
                ],
            )
        elif bundle_id == "compare-systems":
            sys_id, s_sys = _add("systems_rmsd.xvg")
            loaded_summaries = [s_sys]
            spec = FigureSpec(
                journal="nature",
                column_width="one_half",
                aspect_ratio=0.72,
                palette="nature-npg",
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Equilibrated Backbone RMSD Comparison",
                        dataset_ids=[sys_id],
                        plot_type="violin",
                        y_unit="A",
                        equilibration_fraction=0.2,
                        show_pvalues=True,
                        stat_test="mannwhitney",
                    )
                ],
            )
        elif bundle_id == "delta-rmsf":
            rmsf_id, s_rmsf = _add("rmsf_wt_vs_mut.xvg")
            loaded_summaries = [s_rmsf]
            spec = FigureSpec(
                journal="nature",
                column_width="one_half",
                aspect_ratio=0.62,
                nrows=1,
                ncols=1,
                panels=[
                    PanelSpec(
                        id="panel-1",
                        title="Mutation-Induced Flexibility Change (G46A − WT)",
                        dataset_ids=[rmsf_id],
                        plot_type="delta",
                        y_unit="A",
                        domain_tracks=[
                            DomainTrackSpec(start=1, end=38, label="NTD", color="#3B82F6"),
                            DomainTrackSpec(start=39, end=56, label="Loop", color="#E64B35"),
                            DomainTrackSpec(start=57, end=150, label="Catalytic Domain", color="#00A087"),
                        ],
                    )
                ],
            )
        else:
            raise KeyError(f"Unknown sample bundle: {bundle_id}")

        return {"datasets": loaded_summaries, "spec": spec.model_dump(mode="json")}

    def render(self, spec: FigureSpec | dict[str, Any], fmt: str = "svg", dpi: int | None = None) -> bytes:
        fig_spec = spec if isinstance(spec, FigureSpec) else FigureSpec.model_validate(spec)
        return render_figure_bytes(fig_spec, self.store.all_map(), fmt=fmt, dpi=dpi)

    def codegen(self, spec: FigureSpec | dict[str, Any], output_name: str = "figure.pdf") -> dict[str, Any]:
        fig_spec = spec if isinstance(spec, FigureSpec) else FigureSpec.model_validate(spec)
        fmap = self.store.filenames_map()
        return {
            "cli": generate_cli_command(fig_spec, fmap, output_name=output_name),
            "python": generate_python_script(fig_spec, fmap, output_name=output_name),
            "recipe": generate_recipe_dict(fig_spec, fmap),
        }

    def stats(self, dataset_ids: list[str], eq_fraction: float = 0.0) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for ds_id in dataset_ids:
            ds = self.store.get(ds_id)
            if ds is None or ds.kind != "xy":
                continue
            for s in ds.series:
                st = s.summary_stats(eq_fraction=eq_fraction)
                conv = compute_convergence(s.x, s.y, eq_fraction=eq_fraction)
                st.update(conv.to_summary_dict())
                st["dataset_id"] = ds_id
                st["filename"] = ds.filename
                st["series_index"] = s.series_index
                rows.append(st)
        return {"stats": rows}

    def fel_minima(self, spec: FigureSpec | dict[str, Any]) -> dict[str, Any]:
        """Extract energy minima & gmx trjconv -dump commands for any fel2d panel in spec."""
        fig_spec = spec if isinstance(spec, FigureSpec) else FigureSpec.model_validate(spec)
        panels_out: list[dict[str, Any]] = []
        all_ds = self.store.all_map()

        for panel in fig_spec.panels:
            if panel.plot_type != "fel2d":
                continue
            style_lookup = {(st.dataset_id, st.series_index): st for st in panel.series_styles}
            active: list[tuple[SeriesData, ParsedDataset]] = []
            for did in panel.dataset_ids:
                ds = all_ds.get(did)
                if ds is None or ds.kind != "xy":
                    continue
                for s in ds.series:
                    st = style_lookup.get((did, s.series_index))
                    if st is not None and not st.visible:
                        continue
                    xs = unit_scale(ds.x_unit, panel.x_unit)
                    ys = unit_scale(ds.y_unit, panel.y_unit)
                    active.append(
                        (
                            SeriesData(
                                name=(st.label if st and st.label else s.name),
                                x=s.x * xs,
                                y=s.y * ys,
                                source_file=s.source_file,
                                series_index=s.series_index,
                            ),
                            ds,
                        )
                    )
            if len(active) < 2:
                continue
            try:
                tu = panel.x_unit if panel.x_unit != "original" else active[0][1].x_unit
                fel = compute_fel_from_series(
                    active[0][0],
                    active[1][0],
                    temp_k=panel.fel_temp_k,
                    bins=panel.fel_bins,
                    unit=panel.fel_unit,
                    eq_fraction=panel.equilibration_fraction,
                    time_unit=tu,
                )
                panels_out.append(
                    {
                        "panel_id": panel.id,
                        "title": panel.title or panel.id,
                        "unit": fel.unit,
                        "temp_k": fel.temp_k,
                        "x_name": active[0][0].name,
                        "y_name": active[1][0].name,
                        "time_unit": tu or "ps",
                        "minima": [m.to_dict() for m in fel.minima],
                    }
                )
            except Exception:
                continue

        return {"fel_panels": panels_out}
