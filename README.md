# GMXgrace

[![CI](https://github.com/pritampanda15/GMXgrace/actions/workflows/ci.yml/badge.svg)](https://github.com/pritampanda15/GMXgrace/actions/workflows/ci.yml)
[![Online Studio](https://img.shields.io/badge/Online%20Studio-GitHub%20Pages%20(Live)-1d4ed8.svg)](https://pritampanda15.github.io/GMXgrace/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Academic Non-Commercial](https://img.shields.io/badge/License-Academic%20Non--Commercial-orange.svg)](LICENSE)

**Publication-quality GROMACS plotting suite** — a modern Python CLI, Python API, and interactive **Online Web Studio** replacement for `xmgrace`.

**Try Online (Free & Private, runs locally in your browser via Pyodide WebAssembly):**  
👉 **https://pritampanda15.github.io/GMXgrace/** *(your trajectory files never leave your computer)*

One unified Matplotlib rendering engine powers both the command line and the browser studio, so figures crafted interactively online and figures scripted on an HPC cluster are **100% identical**.

---

## Key Features

- **Native GROMACS File Support**
  - **`.xvg` (Grace format)**: Parses `@ title`, `@ subtitle`, axis labels, `@ sN legend`, `@TYPE xy|xydy|xydx|xydxdy`, multi-block `&` sets, `-nxy` multi-column sets, trailing residue annotations (`gmx rama`), and translates Grace escape sequences (`\s...\N`, `\S...\N`, `\f{Symbol}...\f{}`, `\cE\C` → Å).
  - **`.xpm` (2D Matrices)**: Continuous matrices (`gmx sham` Gibbs Free Energy Landscapes with contour isolines, contact/covariance maps) and discrete matrices (`gmx do_dssp` secondary structure timelines with categorical colorbars).
  - **`.edr` (Binary Energy Files)**: Direct parsing via `pyedr`/`panedr` with column selection (`--columns Potential,Temperature`).
  - **Tabular files**: `.csv`, `.tsv`, `.dat`, `.txt` with automatic delimiter and header inference.
- **Publication Journal Presets**
  - `nature` (89 / 120 / 183 mm), `science` (75 / 120 / 175 mm), `acs` (JCTC / JACS / JCIM: 84.6 / 177.8 mm), `biophysj`, `pnas`, and `grace` (modernized `xmgrace`).
  - Colorblind-safe palettes: `okabe-ito`, `nature-npg`, `acs-classic`, `tol-bright`, `tol-muted`, `viridis-discrete`, `grayscale-pub`.
  - Editable vector exports (`PDF` Type-42, `SVG` editable text, `EPS`) and journal-ready high-DPI raster exports (`PNG` & LZW-compressed `TIFF` at 300/600/1200 DPI).
- **MD Analysis Built-In**
  - Automatic unit conversion (`ps ↔ ns ↔ µs`, `nm ↔ Å`, `nm² ↔ Å²`, `kJ/mol ↔ kcal/mol`) with auto-updated axis labels.
  - Replicate aggregation across independent trajectories (`mean_sd`, `mean_sem`, `median_iqr` shaded confidence bands).
  - Running averages & smoothing (`moving_avg`, `savgol`, `gaussian`) with optional faint raw trajectory overlay.
  - Marginal distribution histogram + KDE on the right y-axis.
  - 2D KDE contour + scatter mode (`density2d`) for Ramachandran (`gmx rama`) and PCA 2D projections (`gmx anaeig -2d`).
  - Summary statistics (full and post-equilibration Mean ± SD, Min, Max, Median) with 1-click CSV export for manuscript Supplementary Tables.
- **Online Web Studio + Full Reproducibility**
  - Drag-and-drop browser studio with 6 built-in GROMACS demo showcases, multi-panel layout designer (`1×1`, `1×2`, `2×2`, `2×3`), live SVG preview, per-series styling, annotations, and 1-click vector/raster exports.
  - Live **Reproduce (CLI & Python)** pane generating the exact `gmxgrace` CLI command, standalone Python script, and portable `figure_recipe.json` (both downloadable and re-importable!).

---

## Installation

```bash
git clone https://github.com/pritampanda15/GMXgrace.git
cd GMXgrace

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[edr]"

# Or pip
pip install -e ".[edr]"
```

---

## Command-Line Usage (`gmxgrace`)

### 1. Drop-in `xmgrace`-style plotting

```bash
# Single file -> publication PDF (Nature single-column default)
gmxgrace examples/rmsd_rep1.xvg --x-unit ns --y-unit A -o rmsd.pdf

# Familiar xmgrace -nxy multi-column plotting
gmxgrace -nxy examples/wham_histo.xvg --journal acs --no-legend -o histo.pdf
```

### 2. Replicate Averaging (Mean ± SD) + Marginal KDE

```bash
gmxgrace plot examples/rmsd_rep1.xvg examples/rmsd_rep2.xvg examples/rmsd_rep3.xvg \
  --aggregate mean_sd --show-replicates \
  --x-unit ns --y-unit A --marginal --eq 0.2 --mean-line \
  --journal nature --width one_half --stats \
  -o rmsd_triplicate.pdf
```

### 3. Multi-Panel 2×2 Nature Publication Figure

```bash
gmxgrace plot examples/rmsd_rep1.xvg examples/rmsf_wt_vs_mut.xvg examples/gyrate.xvg examples/gibbs_fel.xpm \
  --grid 2x2 --panel-labels lowercase \
  --x-unit ns --y-unit A \
  --journal nature --width double \
  -o figure1_nature.pdf
```

### 4. GROMACS `.xpm` Matrices & Ramachandran Density Plots

```bash
# Gibbs Free Energy Landscape (gmx sham)
gmxgrace matrix examples/gibbs_fel.xpm --cmap RdYlBu_r --contours 10 --journal acs -o fel.pdf

# Secondary Structure Timeline (gmx do_dssp)
gmxgrace matrix examples/dssp.xpm --no-contours --journal biophysj -o dssp.tiff --dpi 600

# Ramachandran 2D KDE Density (gmx rama)
gmxgrace plot examples/rama.xvg --plot-type density2d --cmap Blues --journal nature -o rama.pdf
```

### 5. Inspect Metadata & Equilibrated Statistics

```bash
gmxgrace inspect examples/rmsd_rep1.xvg examples/rmsf_wt_vs_mut.xvg --eq 0.2
```

### 6. Reproduce Any Online Studio Figure from CLI

```bash
gmxgrace render figure_recipe.json -o final_figure.pdf --dpi 600
```

---

## Python API

```python
import gmxgrace as gg

gg.plot(
    ["examples/rmsd_rep1.xvg", "examples/rmsd_rep2.xvg", "examples/rmsd_rep3.xvg"],
    output="rmsd_pub.pdf",
    journal="nature",
    x_unit="ns",
    y_unit="A",
    aggregate="mean_sd",
    marginal=True,
)
```

---

## Online Web Studio

Start locally (or on an HPC node with SSH port forwarding):

```bash
gmxgrace serve --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

### Deploy Online (Docker / Render / Cloud)

```bash
# Local or server Docker deployment
docker compose up --build

# Render.com
# Connect this repository and deploy directly from render.yaml
```

---

## Citation

If you use **GMXgrace** to prepare figures or analyze GROMACS simulations in an academic publication, please cite it using GitHub's **"Cite this repository"** button or [CITATION.cff](CITATION.cff):

```bibtex
@software{panda2026gmxgrace,
  author       = {Panda, Pritam Kumar},
  title        = {{GMXgrace: Publication-Quality GROMACS Plotting CLI, API, and Online Studio}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/pritampanda15/GMXgrace}
}
```

---

## License (Academic & Non-Commercial Research Only)

Copyright © 2026 **Pritam Kumar Panda**.

This software is licensed strictly under the **[GMXgrace Academic & Non-Commercial Research License (v1.0)](LICENSE)**.

- **Permitted**: Free use, modification, redistribution, and publication by accredited universities, academic labs, students, non-profit research institutes, and public government research laboratories for non-commercial research and education.
- **Prohibited**: Any commercial use, commercial R&D, commercial redistribution, or paid/commercial SaaS hosting without a separate written commercial license from the author (`pritampkp15@gmail.com`).
