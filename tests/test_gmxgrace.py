"""Comprehensive test suite for gmxgrace parsers, analysis, rendering, CLI, and FastAPI studio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gmxgrace import FigureSpec, PanelSpec, parse_bytes, parse_file, render_figure_bytes
from gmxgrace.analysis import aggregate_replicates, smooth_series, unit_scale, update_label_unit
from gmxgrace.cli import main as cli_main
from gmxgrace.codegen import generate_cli_command, generate_python_script
from gmxgrace.grace_text import grace_to_matplotlib
from gmxgrace.parsers.tabular import parse_tabular_text
from gmxgrace.parsers.xpm import parse_xpm_text
from gmxgrace.parsers.xvg import parse_xvg_text
from gmxgrace.samples import get_sample_files, write_sample_datasets
from gmxgrace.web.app import app


def test_grace_escape_sequences() -> None:
    assert "ΔG" in grace_to_matplotlib(r"\f{Symbol}D\f{}G (kJ/mol)")
    assert "nm²" in grace_to_matplotlib(r"Area (nm\S2\N)")
    assert "Å" in grace_to_matplotlib(r"Distance (\cE\C)")
    assert "ξ" in grace_to_matplotlib(r"\f{Symbol}x\f{} (nm)")
    rg = grace_to_matplotlib(r"R\sg\N (nm)")
    assert "R" in rg and "g" in rg


def test_parse_xvg_xy_xydy_and_multiblock() -> None:
    samples = get_sample_files()
    ds = parse_xvg_text(samples["rmsd_rep1.xvg"], filename="rmsd_rep1.xvg")
    assert ds.file_format == "xvg"
    assert ds.detected_type == "rmsd"
    assert ds.x_unit == "ps"
    assert ds.y_unit == "nm"
    assert len(ds.series) == 1
    assert len(ds.series[0].x) == 501

    pmf = parse_xvg_text(samples["pmf_profile.xvg"], filename="pmf_profile.xvg")
    assert pmf.detected_type == "pmf"
    assert len(pmf.series) == 1
    assert pmf.series[0].dy is not None
    assert len(pmf.series[0].dy) == len(pmf.series[0].y)

    # Multi-block & separated sets
    multi_xvg = """
@ title "Multi"
@ xaxis label "Time (ns)"
@ yaxis label "Val (nm)"
@ s0 legend "Set A"
@ s1 legend "Set B"
0.0 0.1
1.0 0.2
&
0.0 0.3
1.0 0.4
&
"""
    mds = parse_xvg_text(multi_xvg, filename="multi.xvg")
    assert len(mds.series) == 2
    assert mds.series[0].name == "Set A"
    assert mds.series[1].name == "Set B"


def test_parse_xpm_continuous_and_discrete() -> None:
    samples = get_sample_files()
    fel = parse_xpm_text(samples["gibbs_fel.xpm"], filename="gibbs_fel.xpm")
    assert fel.kind == "matrix"
    assert fel.detected_type == "fel"
    assert fel.matrix is not None
    assert not fel.matrix.is_discrete
    assert fel.matrix.values.shape == (48, 48)
    assert np.nanmin(fel.matrix.values) >= 0.0

    dssp = parse_xpm_text(samples["dssp.xpm"], filename="dssp.xpm")
    assert dssp.kind == "matrix"
    assert dssp.detected_type == "dssp"
    assert dssp.matrix is not None
    assert dssp.matrix.is_discrete
    assert "A-Helix" in dssp.matrix.categories
    assert dssp.matrix.values.shape == (60, 100)


def test_parse_tabular_csv_and_dat() -> None:
    csv_txt = "# comment\nTime (ns),RMSD (nm),Rg (nm)\n0,0.0,1.8\n1,0.2,1.9\n2,0.25,1.85\n"
    ds = parse_tabular_text(csv_txt, filename="traj.csv")
    assert ds.file_format == "tabular"
    assert len(ds.series) == 2
    assert ds.series[0].name == "RMSD (nm)"
    assert ds.x_unit == "ns"


def test_units_smoothing_and_aggregation() -> None:
    assert unit_scale("ps", "ns") == pytest.approx(1e-3)
    assert unit_scale("nm", "A") == pytest.approx(10.0)
    assert unit_scale("kJ/mol", "kcal/mol") == pytest.approx(1.0 / 4.184)
    assert update_label_unit("Time (ps)", "ps", "ns") == "Time (ns)"
    assert update_label_unit("RMSD (nm)", "nm", "A") == "RMSD (Å)"

    samples = get_sample_files()
    s1 = parse_xvg_text(samples["rmsd_rep1.xvg"]).series[0]
    s2 = parse_xvg_text(samples["rmsd_rep2.xvg"]).series[0]
    s3 = parse_xvg_text(samples["rmsd_rep3.xvg"]).series[0]

    for m in ("moving_avg", "savgol", "gaussian"):
        sm = smooth_series(s1.y, method=m, window=15)
        assert sm.shape == s1.y.shape
        assert np.all(np.isfinite(sm))

    for agg_m in ("mean_sd", "mean_sem", "median_iqr"):
        res = aggregate_replicates([s1, s2, s3], method=agg_m)
        assert res is not None
        assert res.n_replicates == 3
        assert np.all(res.upper >= res.lower)


def test_render_all_export_formats_and_multipanel() -> None:
    samples = get_sample_files()
    datasets = {
        "r1": parse_bytes(samples["rmsd_rep1.xvg"].encode(), "rmsd_rep1.xvg"),
        "r2": parse_bytes(samples["rmsd_rep2.xvg"].encode(), "rmsd_rep2.xvg"),
        "fel": parse_bytes(samples["gibbs_fel.xpm"].encode(), "gibbs_fel.xpm"),
    }
    spec = FigureSpec(
        journal="nature",
        column_width="double",
        nrows=1,
        ncols=2,
        panels=[
            PanelSpec(id="p1", dataset_ids=["r1", "r2"], x_unit="ns", y_unit="A", aggregate="mean_sd", marginal_hist=True),
            PanelSpec(id="p2", dataset_ids=["fel"], plot_type="matrix", show_contours=True),
        ],
    )
    for fmt, magic in [("svg", b"<svg"), ("pdf", b"%PDF"), ("eps", b"%!PS-Adobe"), ("png", b"\x89PNG"), ("tiff", b"II")]:
        raw = render_figure_bytes(spec, datasets, fmt=fmt, dpi=150)
        assert len(raw) > 500
        assert magic in raw[:512] or (fmt == "tiff" and raw[:2] in {b"II", b"MM"})


def test_cli_workflow_and_recipe_reproducibility(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    assert cli_main(["demo", "--out-dir", str(demo_dir)]) == 0
    assert (demo_dir / "rmsd_rep1.xvg").exists()
    assert (demo_dir / "gibbs_fel.xpm").exists()

    # xmgrace-style shorthand
    out_pdf = tmp_path / "rmsd_short.pdf"
    assert cli_main([str(demo_dir / "rmsd_rep1.xvg"), "--x-unit", "ns", "--y-unit", "A", "-o", str(out_pdf)]) == 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 1000

    # Multi-file aggregate + recipe save + render from recipe
    out_agg = tmp_path / "rmsd_agg.png"
    recipe_path = tmp_path / "recipe.json"
    rc = cli_main([
        "plot",
        str(demo_dir / "rmsd_rep1.xvg"),
        str(demo_dir / "rmsd_rep2.xvg"),
        str(demo_dir / "rmsd_rep3.xvg"),
        "--aggregate", "mean_sd",
        "--marginal",
        "--journal", "acs",
        "--save-recipe", str(recipe_path),
        "-o", str(out_agg),
    ])
    assert rc == 0
    assert out_agg.exists()
    assert recipe_path.exists()

    out_repro = tmp_path / "reproduced.svg"
    assert cli_main(["render", str(recipe_path), "-o", str(out_repro)]) == 0
    assert out_repro.exists() and "<svg" in out_repro.read_text(encoding="utf-8")

    # Matrix subcommand & inspect subcommand
    out_fel = tmp_path / "fel.pdf"
    assert cli_main(["matrix", str(demo_dir / "gibbs_fel.xpm"), "--contours", "8", "-o", str(out_fel)]) == 0
    assert out_fel.exists()

    assert cli_main(["inspect", str(demo_dir / "rmsd_rep1.xvg"), str(demo_dir / "dssp.xpm"), "--eq", "0.2"]) == 0
    assert cli_main(["presets"]) == 0

    # Codegen sanity check
    spec = FigureSpec.model_validate(json.loads(recipe_path.read_text())["spec"])
    cli_str = generate_cli_command(spec, {"ds_0": "rmsd_rep1.xvg", "ds_1": "rmsd_rep2.xvg", "ds_2": "rmsd_rep3.xvg"})
    assert "gmxgrace plot" in cli_str and "--aggregate mean_sd" in cli_str
    py_str = generate_python_script(spec, {"ds_0": "rmsd_rep1.xvg"})
    assert "save_figure" in py_str


def test_fastapi_web_studio_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"

    presets = client.get("/api/presets").json()
    assert len(presets["journals"]) >= 6
    assert len(presets["sample_bundles"]) >= 5

    # Load 4-panel showcase
    bundle = client.post("/api/samples/md-4panel")
    assert bundle.status_code == 200
    bdata = bundle.json()
    assert len(bdata["datasets"]) == 6
    spec = bdata["spec"]

    # Render SVG & PDF
    svg_resp = client.post("/api/render", json={"spec": spec, "fmt": "svg"})
    assert svg_resp.status_code == 200
    assert "<svg" in svg_resp.text

    pdf_resp = client.post("/api/render", json={"spec": spec, "fmt": "pdf"})
    assert pdf_resp.status_code == 200
    assert pdf_resp.content.startswith(b"%PDF")

    # Codegen & Stats
    cg = client.post("/api/codegen", json={"spec": spec, "output_name": "fig1.pdf"}).json()
    assert "gmxgrace" in cg["cli"]
    assert "FigureSpec" in cg["python"]

    st = client.post("/api/stats", json={"dataset_ids": [d["id"] for d in bdata["datasets"]], "eq_fraction": 0.2}).json()
    assert len(st["stats"]) >= 4

    # Load DSSP discrete .xpm showcase and verify SVG renders cleanly
    dssp_bundle = client.post("/api/samples/dssp-xpm")
    assert dssp_bundle.status_code == 200
    dssp_svg = client.post("/api/render", json={"spec": dssp_bundle.json()["spec"], "fmt": "svg"})
    assert dssp_svg.status_code == 200
    assert "<svg" in dssp_svg.text

    # Load Ramachandran 2D KDE showcase (with trailing residue labels in .xvg)
    rama_bundle = client.post("/api/samples/rama-2d")
    assert rama_bundle.status_code == 200
    assert rama_bundle.json()["datasets"][0]["detected_type"] == "ramachandran"
    rama_svg = client.post("/api/render", json={"spec": rama_bundle.json()["spec"], "fmt": "svg"})
    assert rama_svg.status_code == 200
    assert "<svg" in rama_svg.text

    # File upload
    samples = get_sample_files()
    up = client.post(
        "/api/upload",
        files=[("files", ("rmsd_test.xvg", samples["rmsd_rep1.xvg"].encode("utf-8"), "text/plain"))],
        data={"nxy": "false"},
    )
    assert up.status_code == 200
    assert up.json()["datasets"][0]["detected_type"] == "rmsd"


def test_heavy_file_vectorized_parse_and_minmax_downsampling() -> None:
    """Verify fast C-speed parsing and peak-preserving Min-Max downsampling on 200k frames."""
    from gmxgrace.analysis import minmax_downsample

    n = 200_000
    x = np.linspace(0.0, 1000000.0, n)
    y = 0.22 + 0.03 * np.sin(x / 5000.0)
    # Inject a sharp transient spike to ensure Min-Max envelope preserves exact peaks
    y[77777] = 0.95
    y[123456] = 0.01

    lines = ['@ title "Heavy RMSD"', '@ xaxis label "Time (ps)"', '@ yaxis label "RMSD (nm)"']
    # Build fast numeric payload
    body = "\n".join(f"{xv:.2f} {yv:.5f}" for xv, yv in zip(x, y))
    ds = parse_xvg_text("\n".join(lines) + "\n" + body, filename="heavy_1us.xvg")
    assert len(ds.series) == 1
    assert len(ds.series[0].x) == n

    # Peak-preserving downsampling must retain exact global spike min/max while reducing vertices
    xd, yd = minmax_downsample(ds.series[0].x, ds.series[0].y, max_points=8000)
    assert len(xd) <= 8002
    assert float(np.max(yd)) == pytest.approx(0.95, abs=1e-4)
    assert float(np.min(yd)) == pytest.approx(0.01, abs=1e-4)

    # Full render of 200k-frame dataset should produce a compact SVG quickly
    spec = FigureSpec(
        journal="nature",
        max_plot_points=8000,
        panels=[PanelSpec(id="p1", dataset_ids=["heavy"], x_unit="ns", y_unit="A", marginal_hist=True)],
    )
    svg = render_figure_bytes(spec, {"heavy": ds}, fmt="svg")
    assert b"<svg" in svg
    assert len(svg) < 1_500_000  # lightweight publication SVG even for 200k raw frames

