"""Realistic synthetic GROMACS .xvg and .xpm datasets for instant demos and testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _format_xvg(
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    command: str,
    columns: list[np.ndarray],
    legends: list[str] | None = None,
    xy_type: str = "xy",
) -> str:
    lines = [
        "# This file was created by gmxgrace demo generator (GROMACS 2024.3 compatible)",
        f"# Command line: {command}",
        f'@    title "{title}"',
    ]
    if subtitle:
        lines.append(f'@    subtitle "{subtitle}"')
    lines.extend(
        [
            f'@    xaxis  label "{xlabel}"',
            f'@    yaxis  label "{ylabel}"',
            f"@TYPE {xy_type}",
            "@ view 0.15, 0.15, 0.75, 0.85",
            "@ legend on",
            "@ legend box on",
        ]
    )
    if legends:
        for i, leg in enumerate(legends):
            lines.append(f'@ s{i} legend "{leg}"')
    arr = np.column_stack(columns)
    for row in arr:
        lines.append("  ".join(f"{v:12.5f}" for v in row))
    return "\n".join(lines) + "\n"


def generate_rmsd_replicate_xvg(seed: int, label: str, plateau: float = 0.24) -> str:
    """Generate a realistic 100 ns (100000 ps) GROMACS backbone RMSD .xvg file."""
    rng = np.random.default_rng(seed)
    t_ps = np.linspace(0.0, 100000.0, 501)
    # Realistic exponential equilibration + correlated conformational drift + thermal noise
    tau = 8500.0 + seed * 900.0
    base = 0.06 + (plateau - 0.06) * (1.0 - np.exp(-t_ps / tau))
    drift = 0.022 * np.sin(2 * np.pi * t_ps / 42000.0 + seed) + 0.012 * np.cos(2 * np.pi * t_ps / 19000.0 + seed * 0.7)
    noise = rng.normal(0.0, 0.014, size=len(t_ps))
    rmsd = np.clip(base + drift + noise, 0.02, None)
    rmsd[0] = 0.0
    return _format_xvg(
        title="RMSD",
        subtitle="Backbone after lsq fit to Backbone",
        xlabel="Time (ps)",
        ylabel="RMSD (nm)",
        command=f"gmx rms -s md.tpr -f md_rep{seed}.xtc -o rmsd_rep{seed}.xvg -tu ps",
        columns=[t_ps, rmsd],
        legends=[label],
    )


def generate_rmsf_xvg() -> str:
    """Generate a two-series per-residue RMSF .xvg comparing Wild-Type vs Mutant."""
    rng = np.random.default_rng(42)
    res = np.arange(1, 151, dtype=float)
    # Secondary structure baseline + flexible N/C termini + active-site loop peaks
    wt = (
        0.075
        + 0.19 * np.exp(-res / 8.0)
        + 0.16 * np.exp(-(150.0 - res) / 7.0)
        + 0.14 * np.exp(-0.5 * ((res - 46.0) / 5.5) ** 2)
        + 0.11 * np.exp(-0.5 * ((res - 98.0) / 4.2) ** 2)
        + 0.08 * np.exp(-0.5 * ((res - 122.0) / 3.5) ** 2)
        + rng.normal(0.0, 0.008, size=len(res))
    )
    mut = wt + 0.13 * np.exp(-0.5 * ((res - 46.0) / 6.0) ** 2) + rng.normal(0.0, 0.009, size=len(res))
    wt = np.clip(wt, 0.04, None)
    mut = np.clip(mut, 0.04, None)
    return _format_xvg(
        title="RMS fluctuation",
        subtitle="C-alpha per residue",
        xlabel="Residue",
        ylabel="RMSF (nm)",
        command="gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg -res",
        columns=[res, wt, mut],
        legends=["Wild-Type", "G46A Mutant"],
    )


def generate_gyrate_xvg() -> str:
    """Generate a realistic GROMACS gmx gyrate .xvg with Grace subscript escape R\\sg\\N."""
    rng = np.random.default_rng(101)
    t_ps = np.linspace(0.0, 100000.0, 501)
    rg = 1.84 + 0.055 * np.exp(-t_ps / 22000.0) + 0.018 * np.sin(t_ps / 11000.0) + rng.normal(0.0, 0.016, len(t_ps))
    return _format_xvg(
        title="Radius of gyration (total)",
        subtitle="Protein",
        xlabel="Time (ps)",
        ylabel=r"R\sg\N (nm)",
        command="gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg",
        columns=[t_ps, rg],
        legends=[r"R\sg\N (Total)"],
    )


def generate_sasa_xvg() -> str:
    """Generate a realistic GROMACS gmx sasa .xvg with Grace superscript nm\\S2\\N."""
    rng = np.random.default_rng(202)
    t_ps = np.linspace(0.0, 100000.0, 501)
    sasa = 86.5 + 3.2 * (1.0 - np.exp(-t_ps / 15000.0)) + 1.1 * np.cos(t_ps / 14000.0) + rng.normal(0.0, 1.4, len(t_ps))
    return _format_xvg(
        title="Solvent Accessible Surface",
        subtitle="Protein",
        xlabel="Time (ps)",
        ylabel=r"Area (nm\S2\N)",
        command="gmx sasa -s md.tpr -f md.xtc -o sasa.xvg",
        columns=[t_ps, sasa],
        legends=["Total SASA"],
    )


def generate_pmf_xvg() -> str:
    """Generate a realistic gmx wham PMF profile with @TYPE xydy bootstrap error."""
    xi = np.linspace(0.45, 3.20, 140)
    # Binding well near 0.85 nm + transition barrier near 1.65 nm + unbound plateau at 0
    pmf = (
        -38.5 * np.exp(-0.5 * ((xi - 0.88) / 0.24) ** 2)
        + 9.5 * np.exp(-0.5 * ((xi - 1.55) / 0.28) ** 2)
        - 6.2 * np.exp(-0.5 * ((xi - 2.05) / 0.35) ** 2)
    )
    pmf -= pmf[-1]
    err = 0.8 + 1.6 * np.exp(-0.5 * ((xi - 1.45) / 0.45) ** 2) + 0.35 * (xi - 0.45) / 2.75
    return _format_xvg(
        title="Umbrella potential",
        subtitle="Bootstrap PMF (200 replicates)",
        xlabel=r"\f{Symbol}x\f{} (nm)",
        ylabel=r"\f{Symbol}D\f{}G (kJ/mol)",
        command="gmx wham -it tpr-files.dat -if pullf-files.dat -o profile.xvg -bsres bsResult.xvg -nBootstrap 200",
        columns=[xi, pmf, err],
        legends=["Ligand Unbinding PMF"],
        xy_type="xydy",
    )


def generate_wham_histo_xvg() -> str:
    """Generate overlapping umbrella sampling window histograms."""
    xi = np.linspace(0.45, 3.20, 220)
    centers = np.linspace(0.65, 3.00, 12)
    cols = [xi]
    legends = []
    for i, c in enumerate(centers):
        sigma = 0.11 + 0.015 * np.sin(i)
        h = np.exp(-0.5 * ((xi - c) / sigma) ** 2)
        cols.append(h)
        legends.append(f"Win {i + 1}")
    return _format_xvg(
        title="Umbrella histograms",
        subtitle="Window overlap check",
        xlabel=r"\f{Symbol}x\f{} (nm)",
        ylabel="Counts (norm.)",
        command="gmx wham -it tpr-files.dat -if pullf-files.dat -hist histo.xvg",
        columns=cols,
        legends=legends,
    )


def generate_fel_xpm() -> str:
    """Generate a realistic GROMACS gmx sham Gibbs Free Energy Landscape .xpm file."""
    nx, ny = 48, 48
    pc1 = np.linspace(-3.2, 3.4, nx)
    pc2 = np.linspace(-2.6, 2.8, ny)
    xx, yy = np.meshgrid(pc1, pc2)

    # Two distinct conformational basins (Closed active state & Open inactive state) + saddle
    p1 = 1.00 * np.exp(-0.5 * (((xx + 1.25) / 0.65) ** 2 + ((yy + 0.65) / 0.58) ** 2))
    p2 = 0.58 * np.exp(-0.5 * (((xx - 1.35) / 0.72) ** 2 + ((yy - 0.80) / 0.65) ** 2))
    p3 = 0.16 * np.exp(-0.5 * (((xx - 0.05) / 0.55) ** 2 + ((yy - 0.10) / 0.45) ** 2))
    prob = np.clip(p1 + p2 + p3 + 1e-4, 1e-4, None)
    kbt = 2.494  # kJ/mol at 300 K
    fel = -kbt * np.log(prob / np.max(prob))
    fel = np.clip(fel, 0.0, 16.0)

    # Quantize into 32 levels with single-character ASCII pixel codes
    nlevels = 32
    codes = [chr(ord("A") + i) if i < 26 else chr(ord("a") + i - 26) for i in range(nlevels)]
    levels = np.linspace(0.0, 16.0, nlevels)

    # Simple blue -> white -> red hex gradient for XPM color table
    def _hex_col(frac: float) -> str:
        r = int(round(30 + 215 * frac))
        g = int(round(60 + 185 * (1.0 - abs(2 * frac - 1.0))))
        b = int(round(220 * (1.0 - frac) + 35))
        return f"#{r:02X}{g:02X}{b:02X}"

    lines = [
        "/* XPM */",
        "/* Generated by gmx sham (gmxgrace demo) */",
        '/* title:   "Gibbs Energy Landscape" */',
        '/* legend:  "Free Energy (kJ/mol)" */',
        '/* x-label: "PC1 (nm)" */',
        '/* y-label: "PC2 (nm)" */',
        '/* type:    "Continuous" */',
        "static char *gromacs_xpm[] = {",
        f'"{nx} {ny} {nlevels} 1",',
    ]
    for i, code in enumerate(codes):
        lines.append(f'"{code}  c {_hex_col(i / (nlevels - 1))} " /* "{levels[i]:.3f}" */,')

    lines.append("/* x-axis:  " + " ".join(f"{v:.3f}" for v in pc1) + " */")
    lines.append("/* y-axis:  " + " ".join(f"{v:.3f}" for v in pc2) + " */")

    # GROMACS XPM rows go from TOP (y_max) down to BOTTOM (y_min)
    for row_idx in range(ny - 1, -1, -1):
        row_chars = []
        for col_idx in range(nx):
            lvl = int(np.argmin(np.abs(levels - fel[row_idx, col_idx])))
            row_chars.append(codes[lvl])
        comma = "," if row_idx > 0 else ""
        lines.append(f'"{"".join(row_chars)}"{comma}')
    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_dssp_xpm() -> str:
    """Generate a realistic GROMACS gmx do_dssp secondary structure timeline .xpm file."""
    rng = np.random.default_rng(77)
    nt = 100  # 0..100 ns
    nres = 60
    t_ns = np.linspace(0.0, 100.0, nt)
    res = np.arange(1, nres + 1, dtype=float)

    # Standard DSSP categories & colors
    palette = [
        ("~", "#FFFFFF", "Coil"),
        ("E", "#E63946", "B-Sheet"),
        ("B", "#1D3557", "B-Bridge"),
        ("S", "#2A9D8F", "Bend"),
        ("T", "#F4A261", "Turn"),
        ("H", "#264653", "A-Helix"),
        ("G", "#9B5DE5", "3-Helix"),
    ]

    # Build realistic residue secondary structure timeline (matrix[res_idx, time_idx])
    grid = np.full((nres, nt), "~", dtype="<U1")
    for t in range(nt):
        # Helix 1: res 6..20 (stable)
        grid[5:20, t] = "H"
        # Turn: res 21..24
        grid[20:24, t] = "T"
        # Beta-strand 1: res 25..31
        grid[24:31, t] = "E"
        # Flexible loop/bend: res 32..36
        grid[31:36, t] = "S" if rng.random() > 0.35 else "~"
        # Beta-strand 2: res 37..43
        grid[36:43, t] = "E"
        # Helix 2: res 47..56 (partially frays after 60 ns)
        fray = 2 if (t > 60 and rng.random() > 0.3) else 0
        grid[46 : 56 - fray, t] = "H"
        if fray:
            grid[56 - fray : 56, t] = "G" if rng.random() > 0.5 else "T"
        # Slight thermal edge breathing
        if rng.random() < 0.25:
            grid[19, t] = "T"
        if rng.random() < 0.2:
            grid[24, t] = "B"

    lines = [
        "/* XPM */",
        "/* Generated by gmx do_dssp (gmxgrace demo) */",
        '/* title:   "Secondary structure" */',
        '/* legend:  "Structure" */',
        '/* x-label: "Time (ns)" */',
        '/* y-label: "Residue" */',
        '/* type:    "Discrete" */',
        "static char *gromacs_xpm[] = {",
        f'"{nt} {nres} {len(palette)} 1",',
    ]
    for code, color, label in palette:
        lines.append(f'"{code}  c {color} " /* "{label}" */,')
    lines.append("/* x-axis:  " + " ".join(f"{v:.1f}" for v in t_ns) + " */")
    lines.append("/* y-axis:  " + " ".join(f"{int(v)}" for v in res) + " */")

    for r in range(nres - 1, -1, -1):
        comma = "," if r > 0 else ""
        lines.append(f'"{"".join(grid[r, :])}"{comma}')
    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_rama_xvg() -> str:
    """Generate a realistic GROMACS gmx rama Ramachandran .xvg with trailing residue labels."""
    rng = np.random.default_rng(314)
    # Alpha-helix (~50%), Beta-sheet (~42%), Left-handed helix (~8%)
    n_alpha, n_beta, n_left = 260, 220, 40
    phi = np.concatenate([
        rng.normal(-63.0, 14.0, n_alpha),
        rng.normal(-122.0, 22.0, n_beta),
        rng.normal(58.0, 12.0, n_left),
    ])
    psi = np.concatenate([
        rng.normal(-42.0, 14.0, n_alpha),
        rng.normal(134.0, 19.0, n_beta),
        rng.normal(38.0, 14.0, n_left),
    ])
    phi = np.clip(phi, -179.0, 179.0)
    psi = np.clip(psi, -179.0, 179.0)
    residues = ["ALA", "GLY", "VAL", "LEU", "ILE", "SER", "THR", "LYS", "GLU", "ASP"]
    lines = [
        "# This file was created by gmxgrace demo generator (GROMACS 2024.3 compatible)",
        "# Command line: gmx rama -f md.xtc -s md.tpr -o rama.xvg",
        '@    title "Ramachandran Plot"',
        '@    xaxis  label "Phi (deg)"',
        '@    yaxis  label "Psi (deg)"',
        "@TYPE xy",
    ]
    for i, (ph, ps) in enumerate(zip(phi, psi)):
        res = f"{residues[i % len(residues)]}-{(i % 150) + 1}"
        lines.append(f"{ph:10.4f}  {ps:10.4f}  {res}")
    return "\n".join(lines) + "\n"


def generate_system_comparison_xvg() -> str:
    """Generate 3 equilibrated MD conditions (Apo vs Inhibitor A vs Inhibitor B) for violin/box stats."""
    rng = np.random.default_rng(512)
    t_ns = np.linspace(0.0, 100.0, 401)
    apo = 0.28 + 0.038 * np.sin(t_ns / 14.0) + rng.normal(0.0, 0.028, len(t_ns))
    inh_a = 0.19 + 0.018 * np.cos(t_ns / 17.0) + rng.normal(0.0, 0.019, len(t_ns))
    inh_b = 0.15 + 0.014 * np.sin(t_ns / 21.0) + rng.normal(0.0, 0.016, len(t_ns))
    return _format_xvg(
        title="Ligand-Induced Backbone Stabilization",
        subtitle="Apo vs Holo MD trajectories",
        xlabel="Time (ns)",
        ylabel="Backbone RMSD (nm)",
        command="gmx rms -s md.tpr -f md.xtc -o systems_rmsd.xvg",
        columns=[t_ns, np.clip(apo, 0.05, None), np.clip(inh_a, 0.05, None), np.clip(inh_b, 0.05, None)],
        legends=["Apo", "Inhibitor A", "Inhibitor B"],
    )


def get_sample_files() -> dict[str, str]:
    """Return all realistic sample GROMACS files keyed by filename."""
    return {
        "rmsd_rep1.xvg": generate_rmsd_replicate_xvg(1, "Replicate 1", plateau=0.23),
        "rmsd_rep2.xvg": generate_rmsd_replicate_xvg(2, "Replicate 2", plateau=0.25),
        "rmsd_rep3.xvg": generate_rmsd_replicate_xvg(3, "Replicate 3", plateau=0.24),
        "rmsf_wt_vs_mut.xvg": generate_rmsf_xvg(),
        "gyrate.xvg": generate_gyrate_xvg(),
        "sasa.xvg": generate_sasa_xvg(),
        "pmf_profile.xvg": generate_pmf_xvg(),
        "wham_histo.xvg": generate_wham_histo_xvg(),
        "rama.xvg": generate_rama_xvg(),
        "systems_rmsd.xvg": generate_system_comparison_xvg(),
        "gibbs_fel.xpm": generate_fel_xpm(),
        "dssp.xpm": generate_dssp_xpm(),
    }


def write_sample_datasets(out_dir: str | Path) -> list[Path]:
    """Write all sample GROMACS files into `out_dir`."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in get_sample_files().items():
        p = d / name
        p.write_text(content, encoding="utf-8")
        written.append(p)
    return written


SAMPLE_BUNDLES: list[dict[str, Any]] = [
    {
        "id": "md-4panel",
        "name": "4-Panel Nature MD",
        "description": "2×2 publication figure: RMSD (Mean ± SD + KDE), RMSF (WT vs Mut), Rg + running avg, and Gibbs FEL (.xpm)",
    },
    {
        "id": "rmsd-replicates",
        "name": "RMSD (n=3)",
        "description": "3 independent 100 ns MD trajectories converted ps→ns & nm→Å with shaded SD band and marginal KDE",
    },
    {
        "id": "fel-xpm",
        "name": "Gibbs FEL (.xpm)",
        "description": "2D Free Energy Landscape from gmx sham with contour levels and publication colormap",
    },
    {
        "id": "dssp-xpm",
        "name": "DSSP (.xpm)",
        "description": "Discrete per-residue secondary structure timeline across 100 ns trajectory",
    },
    {
        "id": "umbrella-pmf",
        "name": "WHAM PMF",
        "description": "1×2 ACS figure with bootstrap error band (@TYPE xydy) and overlapping WHAM window histograms",
    },
    {
        "id": "rama-2d",
        "name": "Ramachandran",
        "description": "GROMACS gmx rama phi/psi 2D KDE contour + scatter plot with alpha/beta basin density",
    },
    {
        "id": "fel-from-xvg",
        "name": "FEL (2 XVGs)",
        "description": "Compute 2D Gibbs Free Energy Landscape from RMSD + Rg .xvg files with M1/M2 minima and trjconv -dump timestamps",
    },
    {
        "id": "compare-systems",
        "name": "Violin + p-val",
        "description": "Equilibrated Apo vs Inhibitor A vs Inhibitor B comparison with Mann-Whitney U significance brackets",
    },
    {
        "id": "delta-rmsf",
        "name": "ΔRMSF + Domains",
        "description": "Mutant − Wild-Type ΔRMSF difference plot with colored protein domain track along the top",
    },
]
