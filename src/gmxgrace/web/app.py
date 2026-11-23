"""FastAPI backend for the gmxgrace Online Publication Figure Studio."""

from __future__ import annotations

import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gmxgrace import __version__
from gmxgrace.codegen import generate_cli_command, generate_python_script, generate_recipe_dict
from gmxgrace.models import AnnotationSpec, FigureSpec, PanelSpec, ParsedDataset
from gmxgrace.parsers import parse_bytes
from gmxgrace.plotting import render_figure_bytes
from gmxgrace.samples import SAMPLE_BUNDLES, get_sample_files
from gmxgrace.styles import list_presets_metadata

STATIC_DIR = Path(__file__).parent / "static"
MAX_STORED_DATASETS = 250


class DatasetStore:
    """Thread-safe bounded LRU in-memory store for uploaded & demo datasets."""

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


store = DatasetStore()

app = FastAPI(
    title="gmxgrace Studio API",
    description="Publication-quality GROMACS plotting studio & REST API",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class RenderRequest(BaseModel):
    spec: FigureSpec
    fmt: str = "svg"
    dpi: int | None = None
    filename: str = "gmxgrace_figure"


class CodegenRequest(BaseModel):
    spec: FigureSpec
    output_name: str = "figure.pdf"


class StatsRequest(BaseModel):
    dataset_ids: list[str] = Field(default_factory=list)
    eq_fraction: float = Field(default=0.0, ge=0.0, le=0.9)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web Studio UI not found.")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "datasets_loaded": len(store.all_map())}


@app.get("/api/presets")
def get_presets() -> dict[str, Any]:
    meta = list_presets_metadata()
    meta["sample_bundles"] = SAMPLE_BUNDLES
    return meta


@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    nxy: bool = Form(default=False),
) -> dict[str, Any]:
    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for uf in files:
        fname = uf.filename or "unnamed.xvg"
        try:
            raw = await uf.read()
            ds = parse_bytes(raw, filename=fname, nxy_hint=nxy)
            ds_id = store.put(ds)
            summary = ds.to_summary_dict()
            summary["id"] = ds_id
            uploaded.append(summary)
        except Exception as exc:
            errors.append({"filename": fname, "error": str(exc)})

    if not uploaded and errors:
        raise HTTPException(status_code=400, detail={"message": "Failed to parse uploaded file(s)", "errors": errors})

    return {"datasets": uploaded, "errors": errors}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    removed = store.delete(dataset_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return {"deleted": dataset_id}


@app.post("/api/samples/{bundle_id}")
def load_sample_bundle(bundle_id: str) -> dict[str, Any]:
    """Load curated realistic GROMACS demo files + preconfigured publication FigureSpec."""
    samples = get_sample_files()

    def _add(fname: str, nxy: bool = False) -> tuple[str, dict[str, Any]]:
        ds = parse_bytes(samples[fname].encode("utf-8"), filename=fname, nxy_hint=nxy)
        ds_id = store.put(ds)
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
                        AnnotationSpec(kind="vspan", value=0.0, value_end=20.0, text="Eq.", color="#64748B", alpha=0.4)
                    ],
                ),
                PanelSpec(
                    id="panel-2",
                    title="Per-Residue Flexibility",
                    dataset_ids=[rmsf_id],
                    y_unit="A",
                    annotations=[
                        AnnotationSpec(kind="vline", value=46.0, text="G46A loop", color="#E64B35", linestyle="--")
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
                        AnnotationSpec(kind="vspan", value=0.0, value_end=20.0, text="Equilibration", color="#64748B", alpha=0.4)
                    ],
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
    else:
        raise HTTPException(status_code=404, detail=f"Unknown sample bundle: {bundle_id}")

    return {"datasets": loaded_summaries, "spec": spec.model_dump(mode="json")}


@app.post("/api/render")
def render_endpoint(req: RenderRequest) -> Response:
    fmt = req.fmt.lower().lstrip(".")
    mime_map = {
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "eps": "application/postscript",
        "png": "image/png",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }
    if fmt not in mime_map:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    try:
        raw_bytes = render_figure_bytes(req.spec, store.all_map(), fmt=fmt, dpi=req.dpi)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Rendering error: {exc}") from exc

    headers = {}
    if fmt != "svg":
        ext = "tiff" if fmt == "tif" else fmt
        headers["Content-Disposition"] = f'attachment; filename="{req.filename}.{ext}"'
    return Response(content=raw_bytes, media_type=mime_map[fmt], headers=headers)


@app.post("/api/codegen")
def codegen_endpoint(req: CodegenRequest) -> JSONResponse:
    fmap = store.filenames_map()
    return JSONResponse(
        {
            "cli": generate_cli_command(req.spec, fmap, output_name=req.output_name),
            "python": generate_python_script(req.spec, fmap, output_name=req.output_name),
            "recipe": generate_recipe_dict(req.spec, fmap),
        }
    )


@app.post("/api/stats")
def stats_endpoint(req: StatsRequest) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ds_id in req.dataset_ids:
        ds = store.get(ds_id)
        if ds is None or ds.kind != "xy":
            continue
        for s in ds.series:
            st = s.summary_stats(eq_fraction=req.eq_fraction)
            st["dataset_id"] = ds_id
            st["filename"] = ds.filename
            st["series_index"] = s.series_index
            rows.append(st)
    return {"stats": rows}
