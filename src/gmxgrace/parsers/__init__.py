"""Unified parser dispatch for GROMACS (.xvg, .xpm, .edr) and tabular (.csv, .tsv, .dat) files."""

from __future__ import annotations

from pathlib import Path

from gmxgrace.models import ParsedDataset
from gmxgrace.parsers.edr import parse_edr, parse_edr_bytes
from gmxgrace.parsers.tabular import parse_tabular, parse_tabular_text
from gmxgrace.parsers.xpm import parse_xpm, parse_xpm_text
from gmxgrace.parsers.xvg import parse_xvg, parse_xvg_text

SUPPORTED_EXTENSIONS = {".xvg", ".xpm", ".edr", ".csv", ".tsv", ".dat", ".txt"}


def parse_file(
    path: str | Path,
    *,
    nxy_hint: bool = False,
    edr_columns: list[str] | None = None,
) -> ParsedDataset:
    """Auto-detect format by extension (and content fallback) and parse a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    ext = p.suffix.lower()
    if ext == ".xvg":
        return parse_xvg(p, nxy_hint=nxy_hint)
    if ext == ".xpm":
        return parse_xpm(p)
    if ext == ".edr":
        return parse_edr(p, columns=edr_columns)
    if ext in {".csv", ".tsv", ".dat", ".txt"}:
        # Check if a .dat/.txt file actually contains Grace @ headers
        head = p.read_text(encoding="utf-8", errors="replace")[:2048]
        if any(ln.lstrip().startswith("@") for ln in head.splitlines()[:30]):
            return parse_xvg(p, nxy_hint=nxy_hint)
        return parse_tabular(p)

    # Content-based fallback
    raw = p.read_bytes()
    return parse_bytes(raw, filename=p.name, nxy_hint=nxy_hint, edr_columns=edr_columns)


def parse_bytes(
    raw: bytes,
    filename: str,
    *,
    nxy_hint: bool = False,
    edr_columns: list[str] | None = None,
) -> ParsedDataset:
    """Parse raw uploaded file bytes by filename extension and header sniffing."""
    ext = Path(filename).suffix.lower()
    if ext == ".edr":
        return parse_edr_bytes(raw, filename=filename, columns=edr_columns)
    text = raw.decode("utf-8", errors="replace")
    if ext == ".xpm" or "/* XPM */" in text[:256]:
        return parse_xpm_text(text, filename=filename)
    if ext == ".xvg" or any(ln.lstrip().startswith("@") for ln in text.splitlines()[:30]):
        return parse_xvg_text(text, filename=filename, nxy_hint=nxy_hint)
    return parse_tabular_text(text, filename=filename)


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "parse_bytes",
    "parse_edr",
    "parse_edr_bytes",
    "parse_file",
    "parse_tabular",
    "parse_tabular_text",
    "parse_xpm",
    "parse_xpm_text",
    "parse_xvg",
    "parse_xvg_text",
]
