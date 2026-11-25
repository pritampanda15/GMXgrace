"""FastAPI backend for the gmxgrace Online Publication Figure Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gmxgrace import __version__
from gmxgrace.models import FigureSpec
from gmxgrace.session import StudioSession

STATIC_DIR = Path(__file__).parent / "static"

session = StudioSession()
store = session.store

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
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "mode": "fastapi", "datasets_loaded": len(store.all_map())}


@app.get("/api/presets")
def get_presets() -> dict[str, Any]:
    return session.get_presets()


@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    nxy: bool = Form(default=False),
) -> dict[str, Any]:
    payloads: list[tuple[str, bytes]] = []
    for uf in files:
        payloads.append((uf.filename or "unnamed.xvg", await uf.read()))

    result = session.upload_files(payloads, nxy=nxy)
    if not result["datasets"] and result["errors"]:
        raise HTTPException(
            status_code=400,
            detail={"message": "Failed to parse uploaded file(s)", "errors": result["errors"]},
        )
    return result


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    if not session.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return {"deleted": dataset_id}


@app.post("/api/samples/{bundle_id}")
def load_sample_bundle(bundle_id: str) -> dict[str, Any]:
    try:
        return session.load_sample_bundle(bundle_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        raw_bytes = session.render(req.spec, fmt=fmt, dpi=req.dpi)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Rendering error: {exc}") from exc

    headers = {}
    if fmt != "svg":
        ext = "tiff" if fmt == "tif" else fmt
        headers["Content-Disposition"] = f'attachment; filename="{req.filename}.{ext}"'
    return Response(content=raw_bytes, media_type=mime_map[fmt], headers=headers)


@app.post("/api/codegen")
def codegen_endpoint(req: CodegenRequest) -> JSONResponse:
    return JSONResponse(session.codegen(req.spec, output_name=req.output_name))


@app.post("/api/stats")
def stats_endpoint(req: StatsRequest) -> dict[str, Any]:
    return session.stats(req.dataset_ids, eq_fraction=req.eq_fraction)


class FelMinimaRequest(BaseModel):
    spec: FigureSpec


@app.post("/api/fel_minima")
def fel_minima_endpoint(req: FelMinimaRequest) -> dict[str, Any]:
    return session.fel_minima(req.spec)


# Serve static studio files both under /static and / so relative paths work identically
# in local FastAPI and static GitHub Pages deployments.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static-legacy")
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static-root")
