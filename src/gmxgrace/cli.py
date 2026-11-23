"""Command-line interface for gmxgrace (xmgrace-compatible + modern subcommands)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from gmxgrace import __version__
from gmxgrace.models import AnnotationSpec, FigureSpec, PanelSpec, ParsedDataset
from gmxgrace.parsers import parse_file
from gmxgrace.plotting import save_figure
from gmxgrace.styles import JOURNAL_PRESETS, PALETTES

console = Console()

_SUBCOMMANDS = {"plot", "matrix", "inspect", "render", "serve", "presets", "demo"}


def _parse_range(raw: str | None) -> tuple[float | None, float | None]:
    if not raw:
        return None, None
    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected range format 'MIN:MAX', got {raw!r}")
    lo = float(parts[0]) if parts[0].strip() else None
    hi = float(parts[1]) if parts[1].strip() else None
    return lo, hi


def _parse_grid(raw: str | None, n_files: int) -> tuple[int, int]:
    if not raw:
        return 1, 1
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected grid format 'ROWSxCOLS' (e.g. 2x2), got {raw!r}")
    return int(parts[0]), int(parts[1])


def _add_common_style_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default=None, help="Output file path (.pdf, .svg, .eps, .png, .tiff)")
    p.add_argument(
        "--journal",
        "-j",
        default="nature",
        choices=list(JOURNAL_PRESETS.keys()),
        help="Journal publication preset (default: nature)",
    )
    p.add_argument(
        "--width",
        "-w",
        default="single",
        choices=["single", "one_half", "double", "custom"],
        help="Journal column width preset (default: single)",
    )
    p.add_argument("--width-mm", type=float, default=None, help="Custom figure width in mm")
    p.add_argument("--height-mm", type=float, default=None, help="Custom figure height in mm")
    p.add_argument("--aspect", type=float, default=0.72, help="Panel height/width aspect ratio")
    p.add_argument(
        "--palette",
        default="okabe-ito",
        choices=list(PALETTES.keys()),
        help="Colorblind-safe scientific palette",
    )
    p.add_argument("--font-family", choices=["sans-serif", "serif", "monospace"], default=None)
    p.add_argument("--font-size", type=float, default=None, help="Base font size override (pt)")
    p.add_argument("--line-width", type=float, default=None, help="Line width override (pt)")
    p.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG/TIFF export (default: 300)")
    p.add_argument("--transparent", action="store_true", help="Transparent figure background")
    p.add_argument("--title", default="", help="Plot title")
    p.add_argument("--suptitle", default="", help="Overall figure super-title")
    p.add_argument("--xlabel", default=None, help="Custom X-axis label")
    p.add_argument("--ylabel", default=None, help="Custom Y-axis label")
    p.add_argument("--xlim", default=None, help="X-axis range MIN:MAX (e.g. 0:100)")
    p.add_argument("--ylim", default=None, help="Y-axis range MIN:MAX (e.g. 0:0.5)")
    p.add_argument("--x-unit", default="original", choices=["original", "ps", "ns", "us", "nm", "A"])
    p.add_argument("--y-unit", default="original", choices=["original", "nm", "A", "nm2", "A2", "kJ/mol", "kcal/mol"])
    p.add_argument("--hline", action="append", default=[], type=float, help="Add horizontal reference line at Y")
    p.add_argument("--vline", action="append", default=[], type=float, help="Add vertical reference line at X")
    p.add_argument("--save-recipe", default=None, help="Also save reproducible JSON recipe to path")


def _cmd_plot(args: argparse.Namespace) -> int:
    datasets: dict[str, ParsedDataset] = {}
    edr_cols = [c.strip() for c in args.columns.split(",")] if getattr(args, "columns", None) else None

    for i, fp in enumerate(args.files):
        ds = parse_file(fp, nxy_hint=args.nxy, edr_columns=edr_cols)
        datasets[f"ds_{i}"] = ds

    x_min, x_max = _parse_range(args.xlim)
    y_min, y_max = _parse_range(args.ylim)
    annotations = [AnnotationSpec(kind="hline", value=v) for v in args.hline] + [
        AnnotationSpec(kind="vline", value=v) for v in args.vline
    ]

    nrows, ncols = _parse_grid(args.grid, len(args.files))
    panels: list[PanelSpec] = []

    if nrows * ncols > 1:
        # Multi-panel mode: assign one file per panel
        for idx in range(min(nrows * ncols, len(args.files))):
            ds_id = f"ds_{idx}"
            panels.append(
                PanelSpec(
                    id=f"panel-{idx + 1}",
                    dataset_ids=[ds_id],
                    plot_type=args.plot_type,
                    x_unit=args.x_unit,
                    y_unit=args.y_unit,
                    smoothing_method=args.smooth,
                    smoothing_window=args.window,
                    marginal_hist=args.marginal,
                    equilibration_fraction=args.eq,
                    show_mean_line=args.mean_line,
                    grid=args.grid_lines,
                    show_legend=not args.no_legend,
                    legend_loc=args.legend_loc,
                    legend_frame=args.legend_frame,
                    cmap=args.cmap,
                    show_contours=not args.no_contours,
                    contour_levels=args.contours,
                )
            )
    else:
        panels.append(
            PanelSpec(
                id="panel-1",
                title=args.title,
                dataset_ids=list(datasets.keys()),
                plot_type=args.plot_type,
                x_label=args.xlabel,
                y_label=args.ylabel,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                x_unit=args.x_unit,
                y_unit=args.y_unit,
                smoothing_method=args.smooth,
                smoothing_window=args.window,
                show_raw_under_smooth=not args.hide_raw,
                aggregate=args.aggregate,
                aggregate_show_individual=args.show_replicates,
                marginal_hist=args.marginal,
                equilibration_fraction=args.eq,
                show_mean_line=args.mean_line,
                grid=args.grid_lines,
                show_legend=not args.no_legend,
                legend_loc=args.legend_loc,
                legend_frame=args.legend_frame,
                annotations=annotations,
                cmap=args.cmap,
                show_contours=not args.no_contours,
                contour_levels=args.contours,
                colorbar_label=args.cbar_label,
            )
        )

    width_mode = "custom" if args.width_mm else args.width
    spec = FigureSpec(
        journal=args.journal,
        column_width=width_mode,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        aspect_ratio=args.aspect,
        palette=args.palette,
        font_family=args.font_family,
        font_size=args.font_size,
        line_width=args.line_width,
        nrows=nrows,
        ncols=ncols,
        share_x=args.share_x,
        share_y=args.share_y,
        panel_labels=args.panel_labels,
        suptitle=args.suptitle,
        dpi=args.dpi,
        transparent=args.transparent,
        panels=panels,
    )

    out_path = Path(args.output) if args.output else Path(args.files[0]).with_suffix(".pdf")
    saved = save_figure(spec, datasets, out_path)
    console.print(f"[bold green]Saved publication figure:[/bold green] {saved} ({args.journal}, {args.dpi} DPI)")

    if args.stats:
        _print_stats_table(datasets, eq_fraction=args.eq)

    if args.save_recipe:
        from gmxgrace.codegen import generate_recipe_dict

        file_map = {f"ds_{i}": str(fp) for i, fp in enumerate(args.files)}
        recipe = generate_recipe_dict(spec, file_map)
        rp = Path(args.save_recipe)
        rp.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        console.print(f"[cyan]Saved reproducible recipe:[/cyan] {rp}")

    return 0


def _print_stats_table(datasets: dict[str, ParsedDataset], eq_fraction: float = 0.0) -> None:
    table = Table(title=f"GROMACS Series Summary Statistics (eq skip: {int(eq_fraction * 100)}%)")
    table.add_column("File", style="cyan")
    table.add_column("Series", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean ± SD", justify="right", style="green")
    table.add_column("Median", justify="right")
    if eq_fraction > 0:
        table.add_column("Eq Mean ± SD", justify="right", style="yellow")

    for ds in datasets.values():
        if ds.kind == "xy":
            for s in ds.series:
                st = s.summary_stats(eq_fraction=eq_fraction)
                row = [
                    ds.filename,
                    s.name,
                    str(st["count"]),
                    f"{st['min']:.4g}",
                    f"{st['max']:.4g}",
                    f"{st['mean']:.4g} ± {st['std']:.3g}",
                    f"{st['median']:.4g}",
                ]
                if eq_fraction > 0:
                    row.append(f"{st['eq_mean']:.4g} ± {st['eq_std']:.3g}")
                table.add_row(*row)
    console.print(table)


def _cmd_inspect(args: argparse.Namespace) -> int:
    datasets: dict[str, ParsedDataset] = {}
    for i, fp in enumerate(args.files):
        ds = parse_file(fp, nxy_hint=args.nxy)
        datasets[f"ds_{i}"] = ds
        console.print(f"\n[bold cyan]{ds.filename}[/bold cyan]  ({ds.file_format.upper()}, detected: [yellow]{ds.detected_type}[/yellow])")
        if ds.title:
            console.print(f"  Title:    {ds.title}")
        if ds.subtitle:
            console.print(f"  Subtitle: {ds.subtitle}")
        console.print(f"  X-axis:   {ds.x_label}  [dim](unit: {ds.x_unit or 'none'})[/dim]")
        console.print(f"  Y-axis:   {ds.y_label}  [dim](unit: {ds.y_unit or 'none'})[/dim]")
        if ds.gmx_command:
            console.print(f"  Command:  [dim]{ds.gmx_command}[/dim]")
        if ds.kind == "matrix" and ds.matrix is not None:
            m = ds.matrix
            console.print(f"  Matrix:   {m.values.shape[1]} × {m.values.shape[0]} ({'discrete' if m.is_discrete else 'continuous'})")
            if m.categories:
                console.print(f"  Classes:  {', '.join(m.categories)}")
    if any(d.kind == "xy" for d in datasets.values()):
        _print_stats_table(datasets, eq_fraction=args.eq)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    recipe_path = Path(args.recipe)
    data = json.loads(recipe_path.read_text(encoding="utf-8"))
    spec = FigureSpec.model_validate(data["spec"])
    files_map: dict[str, str] = data.get("files", {})
    base_dir = Path(args.data_dir) if args.data_dir else recipe_path.parent

    datasets: dict[str, ParsedDataset] = {}
    for ds_id, fname in files_map.items():
        fp = Path(fname)
        if not fp.exists():
            fp = base_dir / fname
        datasets[ds_id] = parse_file(fp)

    out_path = Path(args.output)
    saved = save_figure(spec, datasets, out_path, dpi=args.dpi)
    console.print(f"[bold green]Rendered figure from recipe:[/bold green] {saved}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    console.print(
        f"[bold cyan]gmxgrace Web Studio[/bold cyan] starting on [underline]http://{args.host}:{args.port}[/underline]"
    )
    uvicorn.run(
        "gmxgrace.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    t = Table(title="Publication Journal Presets")
    t.add_column("ID", style="bold cyan")
    t.add_column("Journal")
    t.add_column("Widths (mm)")
    t.add_column("Font / LW")
    t.add_column("Description")
    for p in JOURNAL_PRESETS.values():
        widths = f"{p.widths_mm['single']} / {p.widths_mm['one_half']} / {p.widths_mm['double']}"
        t.add_row(p.id, p.name, widths, f"{p.font_size} pt / {p.line_width} pt", p.description)
    console.print(t)
    console.print(f"\n[bold]Colorblind-safe palettes:[/bold] {', '.join(PALETTES.keys())}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from gmxgrace.samples import write_sample_datasets

    out_dir = Path(args.out_dir)
    written = write_sample_datasets(out_dir)
    console.print(f"[bold green]Generated {len(written)} realistic GROMACS demo files in[/bold green] {out_dir}:")
    for p in written:
        console.print(f"  • {p.name}")
    console.print(
        f"\nTry:\n  [cyan]gmxgrace {out_dir}/rmsd_rep1.xvg {out_dir}/rmsd_rep2.xvg {out_dir}/rmsd_rep3.xvg "
        f"--aggregate mean_sd --x-unit ns --y-unit A --marginal --journal nature -o rmsd_pub.pdf[/cyan]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmxgrace",
        description="Publication-quality GROMACS plotting suite (CLI, Python API, and Online Web Studio).",
    )
    parser.add_argument("-V", "--version", action="version", version=f"gmxgrace {__version__}")
    sub = parser.add_subparsers(dest="command")

    # plot subcommand
    p_plot = sub.add_parser("plot", help="Plot GROMACS .xvg, .xpm, .edr, or tabular files")
    p_plot.add_argument("files", nargs="+", help="Input GROMACS files (.xvg, .xpm, .edr, .csv, .dat)")
    p_plot.add_argument("-nxy", action="store_true", help="xmgrace compatibility: treat multi-column file as XY sets")
    p_plot.add_argument("--columns", default=None, help="Comma-separated energy term names for .edr files")
    p_plot.add_argument(
        "--plot-type",
        default="auto",
        choices=["auto", "line", "scatter", "density2d", "bar", "hist", "matrix", "contour"],
    )
    p_plot.add_argument("--smooth", default="none", choices=["none", "moving_avg", "savgol", "gaussian"])
    p_plot.add_argument("--window", type=int, default=25, help="Smoothing window size (default: 25)")
    p_plot.add_argument("--hide-raw", action="store_true", help="Hide raw trace beneath smoothed curve")
    p_plot.add_argument("--aggregate", default="none", choices=["none", "mean_sd", "mean_sem", "median_iqr"])
    p_plot.add_argument("--show-replicates", action="store_true", help="Show faint individual replicates under mean band")
    p_plot.add_argument("--marginal", action="store_true", help="Add marginal distribution histogram + KDE")
    p_plot.add_argument("--eq", type=float, default=0.0, help="Equilibration fraction to skip for stats/marginal (0-0.9)")
    p_plot.add_argument("--mean-line", action="store_true", help="Overlay dashed mean horizontal line")
    p_plot.add_argument("--grid", dest="grid", default=None, help="Multi-panel grid ROWSxCOLS (e.g. 2x2)")
    p_plot.add_argument("--grid-lines", action="store_true", help="Show subtle coordinate grid lines")
    p_plot.add_argument("--share-x", action="store_true")
    p_plot.add_argument("--share-y", action="store_true")
    p_plot.add_argument(
        "--panel-labels",
        default="lowercase",
        choices=["none", "lowercase", "uppercase", "parens_lower", "parens_upper"],
    )
    p_plot.add_argument("--x-scale", default="linear", choices=["linear", "log"])
    p_plot.add_argument("--y-scale", default="linear", choices=["linear", "log"])
    p_plot.add_argument("--no-legend", action="store_true")
    p_plot.add_argument("--legend-loc", default="best")
    p_plot.add_argument("--legend-frame", action="store_true")
    p_plot.add_argument("--cmap", default="RdYlBu_r", help="Colormap for .xpm matrices")
    p_plot.add_argument("--contours", type=int, default=10, help="Contour levels for FEL/.xpm matrices")
    p_plot.add_argument("--no-contours", action="store_true", help="Disable contour overlays on .xpm matrices")
    p_plot.add_argument("--cbar-label", default=None, help="Colorbar label override")
    p_plot.add_argument("--stats", action="store_true", help="Print summary statistics table")
    _add_common_style_args(p_plot)
    p_plot.set_defaults(func=_cmd_plot)

    # matrix subcommand (alias of plot with matrix defaults)
    p_mat = sub.add_parser("matrix", help="Plot GROMACS .xpm 2D matrices (FEL, DSSP, contact maps)")
    p_mat.add_argument("files", nargs="+", help="Input .xpm matrix file(s)")
    p_mat.add_argument("-nxy", action="store_true", default=False)
    p_mat.add_argument("--plot-type", default="matrix", choices=["matrix", "contour", "auto"])
    p_mat.add_argument("--cmap", default="RdYlBu_r")
    p_mat.add_argument("--contours", type=int, default=10)
    p_mat.add_argument("--no-contours", action="store_true")
    p_mat.add_argument("--cbar-label", default=None)
    p_mat.add_argument("--grid", dest="grid", default=None)
    p_mat.add_argument("--panel-labels", default="lowercase", choices=["none", "lowercase", "uppercase", "parens_lower", "parens_upper"])
    p_mat.set_defaults(
        smooth="none",
        window=25,
        hide_raw=False,
        aggregate="none",
        show_replicates=False,
        marginal=False,
        eq=0.0,
        mean_line=False,
        grid_lines=False,
        share_x=False,
        share_y=False,
        x_scale="linear",
        y_scale="linear",
        no_legend=False,
        legend_loc="best",
        legend_frame=False,
        stats=False,
        columns=None,
    )
    _add_common_style_args(p_mat)
    p_mat.set_defaults(func=_cmd_plot)

    # inspect subcommand
    p_ins = sub.add_parser("inspect", help="Inspect GROMACS file headers, units, and summary statistics")
    p_ins.add_argument("files", nargs="+")
    p_ins.add_argument("-nxy", action="store_true")
    p_ins.add_argument("--eq", type=float, default=0.0, help="Equilibration fraction to skip (0-0.9)")
    p_ins.set_defaults(func=_cmd_inspect)

    # render subcommand
    p_ren = sub.add_parser("render", help="Reproduce a publication figure from a Web Studio JSON recipe")
    p_ren.add_argument("recipe", help="Path to figure_recipe.json exported from Web Studio or CLI")
    p_ren.add_argument("-o", "--output", default="figure.pdf", help="Output figure path")
    p_ren.add_argument("--data-dir", default=None, help="Directory containing referenced data files")
    p_ren.add_argument("--dpi", type=int, default=None, help="Override export DPI")
    p_ren.set_defaults(func=_cmd_render)

    # serve subcommand
    p_srv = sub.add_parser("serve", help="Launch the interactive gmxgrace Online Web Studio")
    p_srv.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for remote/Docker)")
    p_srv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_srv.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    p_srv.set_defaults(func=_cmd_serve)

    # presets subcommand
    p_pre = sub.add_parser("presets", help="List available journal publication presets and color palettes")
    p_pre.set_defaults(func=_cmd_presets)

    # demo subcommand
    p_demo = sub.add_parser("demo", help="Generate realistic GROMACS sample .xvg/.xpm files for testing")
    p_demo.add_argument("--out-dir", default="gmxgrace_demo", help="Target directory for sample files")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # xmgrace-style shorthand: allow `gmxgrace rmsd.xvg ...` or `gmxgrace -nxy energy.xvg ...`
    if raw_argv and raw_argv[0] not in _SUBCOMMANDS and raw_argv[0] not in {"-h", "--help", "-V", "--version"}:
        raw_argv = ["plot", *raw_argv]

    parser = build_parser()
    if not raw_argv:
        parser.print_help()
        return 0

    args = parser.parse_args(raw_argv)
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
