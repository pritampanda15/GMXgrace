"""gmxgrace: Publication-quality GROMACS plotting suite (CLI, Python API, and Web Studio)."""

from gmxgrace.codegen import generate_cli_command, generate_python_script, generate_recipe_dict
from gmxgrace.models import AnnotationSpec, FigureSpec, MatrixData, PanelSpec, ParsedDataset, SeriesData, SeriesStyleSpec
from gmxgrace.parsers import parse_bytes, parse_file
from gmxgrace.plotting import quick_plot as plot
from gmxgrace.plotting import render_figure_bytes, render_to_figure, save_figure
from gmxgrace.styles import JOURNAL_PRESETS, PALETTES

__version__ = "0.1.0"

__all__ = [
    "AnnotationSpec",
    "FigureSpec",
    "JOURNAL_PRESETS",
    "MatrixData",
    "PALETTES",
    "PanelSpec",
    "ParsedDataset",
    "SeriesData",
    "SeriesStyleSpec",
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
