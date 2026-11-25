"""gmxgrace: Publication-quality GROMACS plotting suite (CLI, Python API, and Web Studio)."""

from gmxgrace.codegen import generate_cli_command, generate_python_script, generate_recipe_dict
from gmxgrace.models import (
    AnnotationSpec,
    DomainTrackSpec,
    FigureSpec,
    MatrixData,
    PanelSpec,
    ParsedDataset,
    SeriesData,
    SeriesStyleSpec,
    ZoomInsetSpec,
)
from gmxgrace.parsers import parse_bytes, parse_file
from gmxgrace.plotting import quick_plot as plot
from gmxgrace.plotting import render_figure_bytes, render_to_figure, save_figure
from gmxgrace.styles import JOURNAL_PRESETS, PALETTES

__version__ = "0.2.0"

__all__ = [
    "AnnotationSpec",
    "DomainTrackSpec",
    "FigureSpec",
    "JOURNAL_PRESETS",
    "MatrixData",
    "PALETTES",
    "PanelSpec",
    "ParsedDataset",
    "SeriesData",
    "SeriesStyleSpec",
    "ZoomInsetSpec",
    "__version__",
    "generate_cli_command",
    "generate_python_script",
    "generate_recipe_dict",
    "parse_bytes",
    "parse_file",
    "plot",
    "render_figure_bytes",
    "render_to_figure",
    "save_figure",
]
