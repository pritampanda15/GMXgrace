/* gmxgrace Online Publication Studio — Frontend Controller */

(() => {
  const state = {
    presets: null,
    datasets: {},       // dataset_id -> summary dict
    activeDemo: null,
    activePanelIdx: 0,
    zoom: "fit",
    lastRecipe: null,
    lastStats: [],
    renderTimer: null,
    renderSeq: 0,
    spec: {
      journal: "nature",
      column_width: "single",
      width_mm: null,
      height_mm: null,
      aspect_ratio: 0.72,
      palette: "okabe-ito",
      font_family: null,
      font_size: null,
      line_width: null,
      nrows: 1,
      ncols: 1,
      share_x: false,
      share_y: false,
      panel_labels: "lowercase",
      suptitle: "",
      dpi: 300,
      transparent: false,
      max_plot_points: 12000,
      panels: [createDefaultPanel("panel-1")],
    },
  };

  function createDefaultPanel(id) {
    return {
      id,
      title: "",
      subtitle: "",
      dataset_ids: [],
      series_styles: [],
      plot_type: "auto",
      x_label: null,
      y_label: null,
      x_min: null,
      x_max: null,
      y_min: null,
      y_max: null,
      x_scale: "linear",
      y_scale: "linear",
      y2_label: null,
      y2_min: null,
      y2_max: null,
      y2_unit: "original",
      x_unit: "original",
      y_unit: "original",
      smoothing_method: "none",
      smoothing_window: 25,
      show_raw_under_smooth: true,
      raw_alpha: 0.25,
      aggregate: "none",
      aggregate_label: "Mean ± SD",
      aggregate_show_individual: false,
      aggregate_color: null,
      show_errorbars: true,
      errorbar_style: "band",
      marginal_hist: false,
      marginal_kde: true,
      marginal_bins: 35,
      equilibration_fraction: 0.0,
      show_mean_line: false,
      fel_temp_k: 300.0,
      fel_bins: 45,
      fel_unit: "kJ/mol",
      fel_mark_minima: true,
      show_pvalues: true,
      stat_test: "mannwhitney",
      convergence_blocks: 5,
      domain_tracks: [],
      zoom_inset: null,
      cmap: "RdYlBu_r",
      show_contours: true,
      contour_levels: 10,
      colorbar_label: null,
      v_min: null,
      v_max: null,
      matrix_interpolation: "nearest",
      show_legend: true,
      legend_loc: "best",
      legend_ncol: 1,
      legend_frame: false,
      grid: false,
      minor_ticks: true,
      annotations: [],
    };
  }

  const $ = (id) => document.getElementById(id);

  function showToast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.add("hidden"), 2400);
  }

  function getActivePanel() {
    if (state.activePanelIdx >= state.spec.panels.length) {
      state.activePanelIdx = 0;
    }
    return state.spec.panels[state.activePanelIdx];
  }

  function parseOptionalFloat(val) {
    if (val === "" || val === null || val === undefined) return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  }

  // -------------------------------------------------------------------------
  // Dual-Mode Backend Adapter (FastAPI Local Server OR Browser Pyodide WASM)
  // -------------------------------------------------------------------------

  const MIME_MAP = {
    svg: "image/svg+xml",
    pdf: "application/pdf",
    eps: "application/postscript",
    png: "image/png",
    tiff: "image/tiff",
    tif: "image/tiff",
  };

  let backendMode = "fastapi";
  let pyWorker = null;
  let workerMsgId = 0;
  const workerPending = new Map();

  function updateLoader(stage, pct) {
    const st = $("loader-stage");
    const fill = $("loader-fill");
    if (st) st.textContent = stage;
    if (fill && pct != null) fill.style.width = `${pct}%`;
  }

  function workerCall(action, payload = {}) {
    return new Promise((resolve, reject) => {
      const id = ++workerMsgId;
      workerPending.set(id, { resolve, reject });
      pyWorker.postMessage({ id, action, payload });
    });
  }

  async function initBackend() {
    // 1. Check if FastAPI server is active on this origin
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1400);
      const resp = await fetch("./api/health", { signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) {
        const data = await resp.json().catch(() => null);
        if (data && data.status === "ok") {
          backendMode = "fastapi";
          const badge = $("engine-badge");
          if (badge) {
            badge.textContent = "Local Server";
            badge.title = "Connected to local FastAPI Python backend";
          }
          return;
        }
      }
    } catch {
      // Static hosting (GitHub Pages) — fall through to Pyodide WebAssembly engine
    }

    // 2. Boot browser-local Pyodide Web Worker (100% client-side & private)
    backendMode = "pyodide";
    const badge = $("engine-badge");
    if (badge) {
      badge.textContent = "Browser WASM · Private";
      badge.classList.add("wasm");
      badge.title = "Running Python/Matplotlib locally in your browser via Pyodide WebAssembly — your files never leave your machine";
    }
    setRenderStatus("busy", "Loading WASM Python…");
    updateLoader("Fetching GMXgrace wheel manifest…", 8);

    const manifestResp = await fetch("./wheel_manifest.json");
    if (!manifestResp.ok) {
      throw new Error("Could not find wheel_manifest.json for static Pyodide mode.");
    }
    const manifest = await manifestResp.json();

    pyWorker = new Worker("./pyodide-worker.js");
    pyWorker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") {
        updateLoader(msg.stage, msg.pct);
        setRenderStatus("busy", msg.pct < 100 ? `Loading Python (${msg.pct}%)…` : "Engine ready");
        return;
      }
      const pending = workerPending.get(msg.id);
      if (!pending) return;
      workerPending.delete(msg.id);
      if (msg.ok) pending.resolve(msg.data);
      else pending.reject(new Error(msg.error || "Worker error"));
    };
    pyWorker.onerror = (err) => {
      console.error("Pyodide worker error:", err);
      setRenderStatus("error", "WASM worker failed");
    };

    await workerCall("init", { wheelUrl: manifest.wheel });
  }

  const backend = {
    async getPresets() {
      if (backendMode === "pyodide") return workerCall("presets");
      const resp = await fetch("./api/presets");
      if (!resp.ok) throw new Error(await resp.text());
      return resp.json();
    },

    async loadSample(bundleId) {
      if (backendMode === "pyodide") return workerCall("sample", { bundleId });
      const resp = await fetch(`./api/samples/${bundleId}`, { method: "POST" });
      if (!resp.ok) throw new Error(await resp.text());
      return resp.json();
    },

    async uploadFiles(fileList, nxy) {
      if (backendMode === "pyodide") {
        const files = [];
        for (const f of fileList) {
          const buf = await f.arrayBuffer();
          files.push({ name: f.name, bytes: new Uint8Array(buf) });
        }
        const data = await workerCall("upload", { files, nxy });
        if (!data.datasets.length && data.errors.length) {
          throw new Error(data.errors[0].error || "Parse failed");
        }
        return data;
      }
      const fd = new FormData();
      for (const f of fileList) fd.append("files", f);
      fd.append("nxy", nxy ? "true" : "false");
      const resp = await fetch("./api/upload", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail?.errors?.[0]?.error || data.detail?.message || "Upload failed");
      }
      return data;
    },

    async deleteDataset(datasetId) {
      if (backendMode === "pyodide") return workerCall("delete", { datasetId });
      await fetch(`./api/datasets/${datasetId}`, { method: "DELETE" }).catch(() => {});
    },

    async renderSvg(spec) {
      if (backendMode === "pyodide") {
        const buf = await workerCall("render", { spec, fmt: "svg" });
        return new TextDecoder("utf-8").decode(buf);
      }
      const resp = await fetch("./api/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, fmt: "svg" }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Render failed" }));
        throw new Error(err.detail || "Render error");
      }
      return resp.text();
    },

    async renderBlob(spec, fmt, dpi, filename) {
      if (backendMode === "pyodide") {
        const buf = await workerCall("render", { spec, fmt, dpi });
        return new Blob([buf], { type: MIME_MAP[fmt] || "application/octet-stream" });
      }
      const resp = await fetch("./api/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, fmt, dpi, filename }),
      });
      if (!resp.ok) throw new Error("Export failed");
      return resp.blob();
    },

    async codegen(spec, outputName) {
      if (backendMode === "pyodide") return workerCall("codegen", { spec, outputName });
      const resp = await fetch("./api/codegen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, output_name: outputName }),
      });
      if (!resp.ok) throw new Error("Codegen failed");
      return resp.json();
    },

    async stats(datasetIds, eqFraction) {
      if (backendMode === "pyodide") return workerCall("stats", { datasetIds, eqFraction });
      const resp = await fetch("./api/stats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_ids: datasetIds, eq_fraction: eqFraction }),
      });
      if (!resp.ok) throw new Error("Stats failed");
      return resp.json();
    },

    async felMinima(spec) {
      if (backendMode === "pyodide") return workerCall("fel_minima", { spec });
      const resp = await fetch("./api/fel_minima", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec }),
      });
      if (!resp.ok) throw new Error("FEL minima failed");
      return resp.json();
    },
  };

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  async function init() {
    bindStaticEvents();
    try {
      await initBackend();
      state.presets = await backend.getPresets();
      renderPresetControls();
      // Auto-load the 4-panel Nature MD showcase so the user sees an instant live publication figure
      await loadSampleBundle("md-4panel");
    } catch (err) {
      console.error(err);
      setRenderStatus("error", err.message || "Engine unavailable");
      updateLoader(`Failed to start engine: ${err.message || err}`, 100);
    }
  }

  function renderPresetControls() {
    // Demo pills
    const demoWrap = $("demo-buttons");
    demoWrap.innerHTML = "";
    (state.presets.sample_bundles || []).forEach((b) => {
      const btn = document.createElement("button");
      btn.className = "demo-pill";
      btn.textContent = b.name;
      btn.title = b.description;
      btn.dataset.bundle = b.id;
      btn.addEventListener("click", () => loadSampleBundle(b.id));
      demoWrap.appendChild(btn);
    });

    // Journal cards
    const jWrap = $("journal-cards");
    jWrap.innerHTML = "";
    state.presets.journals.forEach((j) => {
      const card = document.createElement("button");
      card.className = "journal-card";
      card.dataset.journal = j.id;
      card.title = j.description;
      card.innerHTML = `
        <div class="journal-card-title">${j.name}</div>
        <div class="journal-card-sub">${j.widths_mm.single} / ${j.widths_mm.double} mm</div>
      `;
      card.addEventListener("click", () => {
        state.spec.journal = j.id;
        state.spec.palette = j.default_palette;
        state.spec.panel_labels = j.default_panel_labels;
        syncAllControls();
        scheduleRender();
      });
      jWrap.appendChild(card);
    });

    // Palette list
    const pWrap = $("palette-list");
    pWrap.innerHTML = "";
    Object.entries(state.presets.palettes).forEach(([pid, colors]) => {
      const item = document.createElement("div");
      item.className = "palette-item";
      item.dataset.palette = pid;
      const swatches = colors.slice(0, 6).map((c) => `<span class="swatch" style="background:${c}"></span>`).join("");
      item.innerHTML = `<span class="palette-name">${pid}</span><span class="swatches">${swatches}</span>`;
      item.addEventListener("click", () => {
        state.spec.palette = pid;
        syncGlobalControls();
        scheduleRender();
      });
      pWrap.appendChild(item);
    });

    // Colormap dropdown
    const cmapSel = $("p-cmap");
    cmapSel.innerHTML = "";
    state.presets.colormaps.forEach((cm) => {
      const opt = document.createElement("option");
      opt.value = cm;
      opt.textContent = cm;
      cmapSel.appendChild(opt);
    });
  }

  async function loadSampleBundle(bundleId) {
    setRenderStatus("busy", "Loading demo…");
    try {
      const data = await backend.loadSample(bundleId);
      state.datasets = {};
      data.datasets.forEach((ds) => {
        state.datasets[ds.id] = ds;
      });
      state.spec = data.spec;
      state.activePanelIdx = 0;
      state.activeDemo = bundleId;

      document.querySelectorAll(".demo-pill").forEach((el) => {
        el.classList.toggle("active", el.dataset.bundle === bundleId);
      });

      syncAllControls();
      await renderNow();
      showToast(`Loaded demo: ${bundleId}`);
    } catch (err) {
      console.error(err);
      setRenderStatus("error", "Demo load failed");
    }
  }

  // -------------------------------------------------------------------------
  // Upload handling
  // -------------------------------------------------------------------------

  async function handleFiles(fileList) {
    if (!fileList || !fileList.length) return;
    const nxy = $("upload-nxy").checked;
    const statusEl = $("upload-status");
    statusEl.classList.add("hidden");
    setRenderStatus("busy", "Parsing files…");

    try {
      const data = await backend.uploadFiles(fileList, nxy);

      // If user uploads fresh files while on a demo with empty/demo files, attach cleanly to active panel
      if (state.activeDemo) {
        state.activeDemo = null;
        document.querySelectorAll(".demo-pill").forEach((el) => el.classList.remove("active"));
        // Reset to clean 1x1 figure for user's own uploaded files
        state.datasets = {};
        state.spec.nrows = 1;
        state.spec.ncols = 1;
        state.spec.panels = [createDefaultPanel("panel-1")];
        state.activePanelIdx = 0;
      }

      const panel = getActivePanel();
      data.datasets.forEach((ds) => {
        state.datasets[ds.id] = ds;
        if (!panel.dataset_ids.includes(ds.id)) {
          panel.dataset_ids.push(ds.id);
        }
      });

      if (data.errors && data.errors.length) {
        statusEl.textContent = data.errors.map((e) => `${e.filename}: ${e.error}`).join("; ");
        statusEl.classList.remove("hidden");
      }

      syncAllControls();
      await renderNow();
      showToast(`Parsed ${data.datasets.length} GROMACS file(s)`);
    } catch (err) {
      statusEl.textContent = String(err);
      statusEl.classList.remove("hidden");
      setRenderStatus("error", "Upload error");
    }
  }

  // -------------------------------------------------------------------------
  // UI Synchronization
  // -------------------------------------------------------------------------

  function syncAllControls() {
    renderDatasetList();
    renderLayoutControls();
    syncGlobalControls();
    syncActivePanelControls();
    renderSeriesEditor();
    renderAnnotationsList();
    renderDomainsList();
  }

  function renderDatasetList() {
    const wrap = $("dataset-list");
    wrap.innerHTML = "";
    const activePanel = getActivePanel();
    const ids = Object.keys(state.datasets);

    if (!ids.length) {
      wrap.innerHTML = `<div class="help-text">No files uploaded yet.</div>`;
      return;
    }

    ids.forEach((id) => {
      const ds = state.datasets[id];
      const inPanel = activePanel.dataset_ids.includes(id);
      const item = document.createElement("div");
      item.className = `dataset-item ${inPanel ? "in-panel" : ""}`;

      const badgeClass = ds.kind === "matrix" ? "type-badge matrix" : "type-badge";
      const countText = ds.kind === "matrix"
        ? `${ds.matrix.shape[1]}×${ds.matrix.shape[0]}`
        : `${ds.series.length} series`;

      item.innerHTML = `
        <div class="dataset-info">
          <div class="dataset-name" title="${ds.filename}">${ds.filename}</div>
          <div class="dataset-meta">
            <span class="${badgeClass}">${ds.detected_type}</span>
            <span>${countText}</span>
          </div>
        </div>
        <div class="dataset-actions">
          <button class="icon-btn" data-toggle="${id}" title="${inPanel ? "Remove from active panel" : "Add to active panel"}">
            ${inPanel ? "✓" : "+"}
          </button>
          <button class="icon-btn danger" data-del="${id}" title="Delete dataset">×</button>
        </div>
      `;

      item.querySelector("[data-toggle]").addEventListener("click", () => {
        const p = getActivePanel();
        if (p.dataset_ids.includes(id)) {
          p.dataset_ids = p.dataset_ids.filter((x) => x !== id);
        } else {
          p.dataset_ids.push(id);
        }
        syncAllControls();
        scheduleRender();
      });

      item.querySelector("[data-del]").addEventListener("click", async () => {
        await backend.deleteDataset(id);
        delete state.datasets[id];
        state.spec.panels.forEach((p) => {
          p.dataset_ids = p.dataset_ids.filter((x) => x !== id);
          p.series_styles = p.series_styles.filter((s) => s.dataset_id !== id);
        });
        syncAllControls();
        scheduleRender();
      });

      wrap.appendChild(item);
    });
  }

  function renderLayoutControls() {
    const gridKey = `${state.spec.nrows}x${state.spec.ncols}`;
    document.querySelectorAll(".layout-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.grid === gridKey);
    });
    $("panel-labels-select").value = state.spec.panel_labels;

    // Panel tabs
    const sel = $("panel-selector");
    sel.innerHTML = "";
    const letters = "abcdefghijklmnopqrstuvwxyz";
    state.spec.panels.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.className = `panel-tab-btn ${idx === state.activePanelIdx ? "active" : ""}`;
      btn.textContent = `Panel ${letters[idx] || idx + 1} (${p.dataset_ids.length})`;
      btn.addEventListener("click", () => {
        state.activePanelIdx = idx;
        syncAllControls();
      });
      sel.appendChild(btn);
    });

    // Chips of datasets assigned to active panel
    const chips = $("panel-dataset-chips");
    chips.innerHTML = "";
    const p = getActivePanel();
    if (!p.dataset_ids.length) {
      chips.innerHTML = `<span class="help-text">Click + on any file above</span>`;
    } else {
      p.dataset_ids.forEach((did) => {
        const ds = state.datasets[did];
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.innerHTML = `${ds ? ds.filename : did} <button title="Remove">×</button>`;
        chip.querySelector("button").addEventListener("click", () => {
          p.dataset_ids = p.dataset_ids.filter((x) => x !== did);
          syncAllControls();
          scheduleRender();
        });
        chips.appendChild(chip);
      });
    }
  }

  function syncGlobalControls() {
    document.querySelectorAll(".journal-card").forEach((c) => {
      c.classList.toggle("active", c.dataset.journal === state.spec.journal);
    });
    document.querySelectorAll(".palette-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.palette === state.spec.palette);
    });

    $("ctrl-width").value = state.spec.column_width;
    $("custom-dims-row").classList.toggle("hidden", state.spec.column_width !== "custom");
    $("ctrl-width-mm").value = state.spec.width_mm ?? "";
    $("ctrl-height-mm").value = state.spec.height_mm ?? "";
    $("ctrl-dpi").value = String(state.spec.dpi);
    $("ctrl-aspect").value = String(state.spec.aspect_ratio);
    $("val-aspect").textContent = Number(state.spec.aspect_ratio).toFixed(2);
    $("ctrl-font-family").value = state.spec.font_family ?? "";
    $("ctrl-font-size").value = state.spec.font_size ?? "";
    $("ctrl-line-width").value = state.spec.line_width ?? "";
    $("ctrl-transparent").checked = Boolean(state.spec.transparent);
    $("paper-sheet")?.classList.toggle("transparent-bg", Boolean(state.spec.transparent));
    $("ctrl-suptitle").value = state.spec.suptitle || "";
    if ($("ctrl-max-pts")) {
      $("ctrl-max-pts").value = String(state.spec.max_plot_points ?? 12000);
    }

    updateDimensionsReadout();
  }

  function updateDimensionsReadout() {
    if (!state.presets) return;
    const j = state.presets.journals.find((x) => x.id === state.spec.journal) || state.presets.journals[0];
    let w = state.spec.column_width === "custom" && state.spec.width_mm
      ? state.spec.width_mm
      : j.widths_mm[state.spec.column_width] || j.widths_mm.single;
    if (state.spec.ncols >= 2 && state.spec.column_width === "single" && !state.spec.width_mm) {
      w = j.widths_mm.double;
    }
    const h = state.spec.height_mm || Math.round((w / state.spec.ncols) * state.spec.aspect_ratio * state.spec.nrows);
    $("canvas-dims").textContent = `${j.name} · ${Math.round(w)} × ${Math.round(h)} mm · ${state.spec.dpi} DPI`;
  }

  function syncActivePanelControls() {
    const p = getActivePanel();
    $("p-title").value = p.title || "";
    $("p-plot-type").value = p.plot_type || "auto";
    $("p-legend-loc").value = p.legend_loc || "best";
    $("p-x-unit").value = p.x_unit || "original";
    $("p-y-unit").value = p.y_unit || "original";
    $("p-xlabel").value = p.x_label ?? "";
    $("p-ylabel").value = p.y_label ?? "";
    if ($("p-y2label")) $("p-y2label").value = p.y2_label ?? "";
    if ($("p-y2-unit")) $("p-y2-unit").value = p.y2_unit || "original";
    $("p-xmin").value = p.x_min ?? "";
    $("p-xmax").value = p.x_max ?? "";
    $("p-ymin").value = p.y_min ?? "";
    $("p-ymax").value = p.y_max ?? "";
    $("p-xscale").value = p.x_scale || "linear";
    $("p-yscale").value = p.y_scale || "linear";
    $("p-show-legend").checked = p.show_legend !== false;
    $("p-legend-frame").checked = Boolean(p.legend_frame);
    $("p-grid").checked = Boolean(p.grid);
    $("p-minor-ticks").checked = p.minor_ticks !== false;

    // Analysis controls
    $("p-aggregate").value = p.aggregate || "none";
    $("p-agg-label").value = p.aggregate_label || "";
    $("p-agg-indiv").checked = Boolean(p.aggregate_show_individual);
    $("p-smooth").value = p.smoothing_method || "none";
    $("p-window").value = String(p.smoothing_window || 25);
    $("val-window").textContent = String(p.smoothing_window || 25);
    $("p-show-raw").checked = p.show_raw_under_smooth !== false;
    $("p-marginal").checked = Boolean(p.marginal_hist);
    $("p-mean-line").checked = Boolean(p.show_mean_line);
    $("p-eq").value = String(p.equilibration_fraction || 0);
    const eqPct = `${Math.round((p.equilibration_fraction || 0) * 100)}%`;
    $("val-eq").textContent = eqPct;
    $("stats-eq-label").textContent = eqPct;

    // Matrix controls
    $("p-cmap").value = p.cmap || "RdYlBu_r";
    $("p-contours").value = String(p.contour_levels || 10);
    $("p-cbar-label").value = p.colorbar_label ?? "";
    $("p-show-contours").checked = p.show_contours !== false;

    // FEL 2D, Stats/Convergence & Zoom Inset controls
    if ($("p-fel-temp")) $("p-fel-temp").value = String(p.fel_temp_k ?? 300);
    if ($("p-fel-bins")) $("p-fel-bins").value = String(p.fel_bins ?? 45);
    if ($("p-fel-unit")) $("p-fel-unit").value = p.fel_unit || "kJ/mol";
    if ($("p-fel-minima")) $("p-fel-minima").checked = p.fel_mark_minima !== false;
    if ($("p-stat-test")) $("p-stat-test").value = p.stat_test || "mannwhitney";
    if ($("p-conv-blocks")) $("p-conv-blocks").value = String(p.convergence_blocks ?? 5);
    if ($("p-show-pvalues")) $("p-show-pvalues").checked = p.show_pvalues !== false;
    const zin = p.zoom_inset || {};
    if ($("p-inset-enabled")) $("p-inset-enabled").checked = Boolean(zin.enabled);
    if ($("p-inset-xmin")) $("p-inset-xmin").value = zin.x_min ?? "";
    if ($("p-inset-xmax")) $("p-inset-xmax").value = zin.x_max ?? "";
    if ($("p-inset-loc")) $("p-inset-loc").value = zin.loc || "lower right";
  }

  function renderDomainsList() {
    const list = $("dom-list");
    if (!list) return;
    const p = getActivePanel();
    list.innerHTML = "";
    (p.domain_tracks || []).forEach((tr, idx) => {
      const row = document.createElement("div");
      row.className = "ann-item";
      row.innerHTML = `<span><span class="swatch" style="display:inline-block;vertical-align:middle;margin-right:5px;background:${tr.color}"></span>${tr.label} (${tr.start}–${tr.end})</span><button class="icon-btn danger">×</button>`;
      row.querySelector("button").addEventListener("click", () => {
        p.domain_tracks.splice(idx, 1);
        renderDomainsList();
        scheduleRender();
      });
      list.appendChild(row);
    });
  }

  function renderAnnotationsList() {
    const p = getActivePanel();
    const list = $("ann-list");
    list.innerHTML = "";
    (p.annotations || []).forEach((ann, idx) => {
      const row = document.createElement("div");
      row.className = "ann-item";
      const desc = ann.value_end != null
        ? `${ann.kind} [${ann.value}..${ann.value_end}] ${ann.text || ""}`
        : `${ann.kind} @ ${ann.value} ${ann.text || ""}`;
      row.innerHTML = `<span>${desc}</span><button class="icon-btn danger">×</button>`;
      row.querySelector("button").addEventListener("click", () => {
        p.annotations.splice(idx, 1);
        renderAnnotationsList();
        scheduleRender();
      });
      list.appendChild(row);
    });
  }

  function renderSeriesEditor() {
    const p = getActivePanel();
    const wrap = $("series-editor-list");
    wrap.innerHTML = "";
    const pal = (state.presets?.palettes?.[state.spec.palette]) || ["#0072B2", "#D55E00", "#009E73"];
    let colorIdx = 0;

    p.dataset_ids.forEach((did) => {
      const ds = state.datasets[did];
      if (!ds || ds.kind !== "xy") return;
      ds.series.forEach((s) => {
        const defaultColor = pal[colorIdx % pal.length];
        colorIdx++;
        let st = p.series_styles.find((x) => x.dataset_id === did && x.series_index === s.index);
        const ensureStyle = () => {
          if (!st) {
            st = {
              dataset_id: did,
              series_index: s.index,
              label: s.name,
              visible: true,
              color: defaultColor,
              linewidth: null,
              linestyle: "-",
              marker: "none",
              markersize: 3.5,
              alpha: 1.0,
              fill_under: false,
              y_axis: "left",
            };
            p.series_styles.push(st);
          }
          return st;
        };

        const card = document.createElement("div");
        card.className = "series-card";
        card.innerHTML = `
          <div class="series-card-head">
            <input type="checkbox" class="s-vis" ${(st ? st.visible : true) ? "checked" : ""} title="Show/Hide series" />
            <input type="color" class="s-col" value="${(st && st.color) || defaultColor}" />
            <input type="text" class="s-lbl" value="${(st && st.label) || s.name}" placeholder="Legend label" />
          </div>
          <div class="series-card-grid">
            <select class="s-ls" title="Line style">
              <option value="-">Solid —</option>
              <option value="--">Dash - -</option>
              <option value="-.">DashDot</option>
              <option value=":">Dot ···</option>
              <option value="none">No line</option>
            </select>
            <select class="s-mk" title="Marker">
              <option value="none">No mark</option>
              <option value="o">Circle ●</option>
              <option value="s">Square ■</option>
              <option value="^">Tri ▲</option>
              <option value="D">Diam ◆</option>
            </select>
            <input type="number" class="s-lw" step="0.1" min="0.2" max="5" placeholder="LW" value="${st?.linewidth ?? ""}" title="Line width (pt)" />
          </div>
          <div class="control-row" style="align-items:center">
            <label class="toggle-row">
              <input type="checkbox" class="s-fill" ${st?.fill_under ? "checked" : ""} />
              <span>Fill under</span>
            </label>
            <select class="s-yaxis" title="Assign to Left or Right Y-axis (Twin-Y)">
              <option value="left">Left Y-Axis</option>
              <option value="right">Right Y-Axis</option>
            </select>
          </div>
        `;

        card.querySelector(".s-ls").value = st?.linestyle || "-";
        card.querySelector(".s-mk").value = st?.marker || "none";
        card.querySelector(".s-yaxis").value = st?.y_axis || "left";

        card.querySelector(".s-vis").addEventListener("change", (e) => {
          ensureStyle().visible = e.target.checked;
          scheduleRender();
        });
        card.querySelector(".s-col").addEventListener("input", (e) => {
          ensureStyle().color = e.target.value;
          scheduleRender();
        });
        card.querySelector(".s-lbl").addEventListener("input", (e) => {
          ensureStyle().label = e.target.value;
          scheduleRender();
        });
        card.querySelector(".s-ls").addEventListener("change", (e) => {
          ensureStyle().linestyle = e.target.value;
          scheduleRender();
        });
        card.querySelector(".s-mk").addEventListener("change", (e) => {
          ensureStyle().marker = e.target.value;
          scheduleRender();
        });
        card.querySelector(".s-lw").addEventListener("input", (e) => {
          ensureStyle().linewidth = parseOptionalFloat(e.target.value);
          scheduleRender();
        });
        card.querySelector(".s-fill").addEventListener("change", (e) => {
          ensureStyle().fill_under = e.target.checked;
          scheduleRender();
        });
        card.querySelector(".s-yaxis").addEventListener("change", (e) => {
          ensureStyle().y_axis = e.target.value;
          scheduleRender();
        });

        wrap.appendChild(card);
      });
    });

    if (!wrap.children.length) {
      wrap.innerHTML = `<div class="help-text">No XY series in the active panel (matrix .xpm files are configured in the Analysis tab).</div>`;
    }
  }

  // -------------------------------------------------------------------------
  // Live Rendering, Codegen & Stats
  // -------------------------------------------------------------------------

  function setRenderStatus(kind, text) {
    const el = $("render-indicator");
    el.className = `render-badge ${kind === "ok" ? "" : kind}`;
    el.textContent = text;
  }

  function scheduleRender() {
    clearTimeout(state.renderTimer);
    state.renderTimer = setTimeout(() => renderNow(), 160);
  }

  async function renderNow() {
    const seq = ++state.renderSeq;
    setRenderStatus("busy", "Rendering…");
    updateDimensionsReadout();

    try {
      const hasFel2d = state.spec.panels.some((p) => p.plot_type === "fel2d");
      const [svgRes, codeRes, statsRes, felRes] = await Promise.allSettled([
        backend.renderSvg(state.spec),
        backend.codegen(state.spec, "publication_figure.pdf"),
        backend.stats(Object.keys(state.datasets), getActivePanel().equilibration_fraction || 0),
        hasFel2d ? backend.felMinima(state.spec) : Promise.resolve({ fel_panels: [] }),
      ]);

      if (seq !== state.renderSeq) return; // stale response

      if (svgRes.status === "rejected") {
        const msg = svgRes.reason?.message || "Render error";
        setRenderStatus("error", msg);
        $("svg-preview").innerHTML = `<div class="empty-canvas" style="color:#dc2626">${msg}</div>`;
      } else {
        $("svg-preview").innerHTML = svgRes.value;
        applyZoom();
        attachSvgHoverInspector();
        setRenderStatus("ok", "Live SVG Ready");
      }

      if (codeRes.status === "fulfilled") {
        const code = codeRes.value;
        $("cli-code").textContent = code.cli;
        $("python-code").textContent = code.python;
        state.lastRecipe = code.recipe;
      }

      if (statsRes.status === "fulfilled") {
        state.lastStats = statsRes.value.stats || [];
        renderStatsTable(state.lastStats);
      }

      if (felRes.status === "fulfilled") {
        renderFelMinimaTable(felRes.value.fel_panels || []);
      }
    } catch (err) {
      if (seq === state.renderSeq) {
        setRenderStatus("error", err.message || "Render error");
      }
    }
  }

  function attachSvgHoverInspector() {
    const svg = $("svg-preview")?.querySelector("svg");
    const readout = $("hover-coords");
    if (!svg || !readout) return;
    const desc = svg.querySelector("#gmxgrace-axes-meta");
    if (!desc) return;
    let metaList = [];
    try {
      metaList = JSON.parse(desc.textContent || "[]");
    } catch {
      return;
    }

    const mapPoint = (e) => {
      for (const m of metaList) {
        const g = svg.querySelector(`#${CSS.escape(m.gid)}`);
        if (!g) continue;
        // First child patch inside a Matplotlib axes group is the exact data-frame box
        const patch = g.querySelector('[id^="patch_"]') || g;
        const rect = patch.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
          const fx = (e.clientX - rect.left) / rect.width;
          const fy = 1.0 - (e.clientY - rect.top) / rect.height;
          let xVal = m.xlim[0] + fx * (m.xlim[1] - m.xlim[0]);
          let yVal = m.ylim[0] + fy * (m.ylim[1] - m.ylim[0]);
          if (m.xscale === "log" && m.xlim[0] > 0 && m.xlim[1] > 0) {
            xVal = Math.pow(10, Math.log10(m.xlim[0]) + fx * (Math.log10(m.xlim[1]) - Math.log10(m.xlim[0])));
          }
          if (m.yscale === "log" && m.ylim[0] > 0 && m.ylim[1] > 0) {
            yVal = Math.pow(10, Math.log10(m.ylim[0]) + fy * (Math.log10(m.ylim[1]) - Math.log10(m.ylim[0])));
          }
          return { xVal, yVal, meta: m };
        }
      }
      return null;
    };

    svg.style.cursor = "crosshair";
    svg.addEventListener("mousemove", (e) => {
      if (e.target.closest && e.target.closest('[id^="gmx_edit__"]')) {
        readout.textContent = "Click to edit label inline";
        return;
      }
      const hit = mapPoint(e);
      if (!hit) {
        readout.textContent = "Click any label to edit · Hover for (x, y)";
        return;
      }
      const xStr = Number.isFinite(hit.xVal) ? hit.xVal.toPrecision(4) : "—";
      const yStr = Number.isFinite(hit.yVal) ? hit.yVal.toPrecision(4) : "—";
      readout.textContent = `x = ${xStr}, y = ${yStr}`;
    });

    svg.addEventListener("mouseleave", () => {
      readout.textContent = "Click any label to edit · Hover for (x, y)";
    });

    // Click-to-edit any tagged Title, X/Y/Right-Y/Colorbar label, Legend item, or Suptitle!
    svg.querySelectorAll('[id^="gmx_edit__"]').forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        openInlineEditor(el);
      });
    });

    svg.addEventListener("click", async (e) => {
      if (e.target.closest && e.target.closest('[id^="gmx_edit__"]')) return;
      const hit = mapPoint(e);
      if (!hit) return;

      // Auto-select clicked panel in multi-panel layouts
      const mIdx = /^gmx_ax_(\d+)$/.exec(hit.meta.gid || "");
      if (mIdx) {
        const pIdx = Number(mIdx[1]);
        if (Number.isFinite(pIdx) && pIdx < state.spec.panels.length && pIdx !== state.activePanelIdx) {
          state.activePanelIdx = pIdx;
          syncAllControls();
        }
      }

      const xStr = hit.xVal.toFixed(3);
      const yStr = hit.yVal.toFixed(3);
      const xl = (hit.meta.xlabel || "").toLowerCase();
      let msg = `Pinned (${xStr}, ${yStr})`;
      let clip = `${xStr}, ${yStr}`;
      if (xl.includes("time")) {
        const psVal = xl.includes("(ns)") ? hit.xVal * 1000 : hit.xVal;
        clip = `gmx trjconv -s md.tpr -f md.xtc -o frame.pdb -dump ${psVal.toFixed(1)}`;
        msg = `Copied: ${clip}`;
      }
      try {
        await navigator.clipboard.writeText(clip);
      } catch {
        // clipboard optional
      }
      showToast(msg);
    });
  }

  let activeInlineCommit = null;

  function closeInlineEditor() {
    const box = $("inline-editor");
    if (box) box.classList.add("hidden");
    activeInlineCommit = null;
  }

  function findNthVisibleSeries(panel, targetIdx) {
    let seen = 0;
    for (const did of panel.dataset_ids || []) {
      const ds = state.datasets[did];
      if (!ds || ds.kind !== "xy") continue;
      for (const s of ds.series || []) {
        const st = (panel.series_styles || []).find((x) => x.dataset_id === did && x.series_index === s.index);
        if (st && !st.visible) continue;
        if (seen === targetIdx) {
          return { datasetId: did, seriesIndex: s.index, defaultName: s.name, existingStyle: st };
        }
        seen++;
      }
    }
    return null;
  }

  function openInlineEditor(svgGroupEl) {
    const gid = svgGroupEl.id || "";
    const parts = gid.split("__"); // ["gmx_edit", field, panelIdx, ...]
    const field = parts[1];
    const panelIdx = parts.length >= 3 ? Number(parts[2]) : state.activePanelIdx;
    const panel = state.spec.panels[panelIdx] || getActivePanel();
    const letters = "abcdefghijklmnopqrstuvwxyz";
    const pLetter = letters[panelIdx] || String(panelIdx + 1);

    // Switch active panel immediately so sidebar stays in sync
    if (field !== "suptitle" && Number.isFinite(panelIdx) && panelIdx < state.spec.panels.length) {
      state.activePanelIdx = panelIdx;
      syncAllControls();
    }

    const currentText = (svgGroupEl.textContent || "").trim();
    let tagLabel = `Panel ${pLetter}`;
    let initialVal = currentText;
    let applyFn = null;

    if (field === "title") {
      tagLabel = `Panel ${pLetter} · Title`;
      initialVal = panel.title || currentText;
      applyFn = (val) => { panel.title = val; };
    } else if (field === "xlabel") {
      tagLabel = `Panel ${pLetter} · X-Axis Label`;
      initialVal = panel.x_label ?? currentText;
      applyFn = (val) => { panel.x_label = val.trim() === "" ? null : val; };
    } else if (field === "ylabel") {
      tagLabel = `Panel ${pLetter} · Y-Axis Label`;
      initialVal = panel.y_label ?? currentText;
      applyFn = (val) => { panel.y_label = val.trim() === "" ? null : val; };
    } else if (field === "y2label") {
      tagLabel = `Panel ${pLetter} · Right Y Label`;
      initialVal = panel.y2_label ?? currentText;
      applyFn = (val) => { panel.y2_label = val.trim() === "" ? null : val; };
    } else if (field === "cbar") {
      tagLabel = `Panel ${pLetter} · Colorbar Label`;
      initialVal = panel.colorbar_label ?? currentText;
      applyFn = (val) => { panel.colorbar_label = val.trim() === "" ? null : val; };
    } else if (field === "agglabel") {
      tagLabel = `Panel ${pLetter} · Aggregate Legend`;
      initialVal = panel.aggregate_label || currentText;
      applyFn = (val) => { panel.aggregate_label = val; };
    } else if (field === "suptitle") {
      tagLabel = "Figure Super-Title";
      initialVal = state.spec.suptitle || currentText;
      applyFn = (val) => { state.spec.suptitle = val; };
    } else if (field === "legend") {
      const legIdx = Number(parts[3] || 0);
      const match = findNthVisibleSeries(panel, legIdx);
      if (!match) return;
      tagLabel = `Panel ${pLetter} · Legend #${legIdx + 1}`;
      initialVal = (match.existingStyle && match.existingStyle.label) || match.defaultName || currentText;
      applyFn = (val) => {
        if (!panel.series_styles) panel.series_styles = [];
        let st = match.existingStyle;
        if (!st) {
          st = {
            dataset_id: match.datasetId,
            series_index: match.seriesIndex,
            label: val,
            visible: true,
            color: null,
            linewidth: null,
            linestyle: "-",
            marker: "none",
            markersize: 3.5,
            alpha: 1.0,
            fill_under: false,
            y_axis: "left",
          };
          panel.series_styles.push(st);
        } else {
          st.label = val;
        }
      };
    } else {
      return;
    }

    const box = $("inline-editor");
    const input = $("inline-editor-input");
    const tag = $("inline-editor-tag");
    if (!box || !input || !tag) return;

    tag.textContent = tagLabel;
    input.value = initialVal;

    // Position cleanly near the clicked SVG label without going offscreen
    const rect = svgGroupEl.getBoundingClientRect();
    box.classList.remove("hidden");
    const bw = box.offsetWidth || 280;
    const bh = box.offsetHeight || 72;
    const left = Math.max(12, Math.min(window.innerWidth - bw - 12, rect.left + rect.width / 2 - bw / 2));
    const top = Math.max(12, Math.min(window.innerHeight - bh - 12, rect.bottom + 8));
    box.style.left = `${Math.round(left)}px`;
    box.style.top = `${Math.round(top)}px`;

    activeInlineCommit = () => {
      applyFn(input.value);
      closeInlineEditor();
      syncAllControls();
      renderNow();
      showToast(`Updated ${tagLabel}`);
    };

    input.focus();
    input.select();
  }

  function renderFelMinimaTable(felPanels) {
    const tbody = $("fel-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    const allRows = [];
    felPanels.forEach((fp) => {
      (fp.minima || []).forEach((m) => allRows.push({ fp, m }));
    });
    if (!allRows.length) {
      tbody.innerHTML = `<tr><td colspan="7" style="color:#64748b">Load "FEL (2 XVGs)" or set Plot Mode → "2D FEL (2 XVGs)" with 2 series to detect energy minima.</td></tr>`;
      return;
    }
    allRows.forEach(({ fp, m }) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${m.label}</strong></td>
        <td>${m.x}</td>
        <td>${m.y}</td>
        <td>${m.dg} ${fp.unit}</td>
        <td>${m.time} ${fp.time_unit}</td>
        <td>#${m.frame_index}</td>
        <td>
          <code style="font-size:11px;margin-right:6px">${m.trjconv_cmd}</code>
          <button class="btn btn-outline btn-sm fel-copy">Copy</button>
        </td>
      `;
      tr.querySelector(".fel-copy").addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(m.trjconv_cmd);
          showToast(`Copied ${m.label} trjconv command!`);
        } catch {
          showToast("Copy blocked by browser");
        }
      });
      tbody.appendChild(tr);
    });
  }

  function formatLabelHtml(text) {
    const escaped = String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return escaped
      .replace(/\$\{_\\mathrm\{([^}]+)\}\}\$/g, "<sub>$1</sub>")
      .replace(/\$\{\^\\mathrm\{([^}]+)\}\}\$/g, "<sup>$1</sup>")
      .replace(/\$\{_\{?\\mathrm\{([^}]+)\}\}?\}\$/g, "<sub>$1</sub>")
      .replace(/\$\{\^\{?\\mathrm\{([^}]+)\}\}?\}\$/g, "<sup>$1</sup>")
      .replace(/\$_([^$]+)\$/g, "<sub>$1</sub>")
      .replace(/\$\^([^$]+)\$/g, "<sup>$1</sup>");
  }

  function renderStatsTable(rows) {
    const tbody = $("stats-tbody");
    tbody.innerHTML = "";
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="11" style="color:#64748b">No XY series loaded.</td></tr>`;
      return;
    }
    const fmt = (v) => (Number.isFinite(v) ? Number(v).toPrecision(4) : "—");
    rows.forEach((r) => {
      const status = r.convergence_status || "—";
      const cls = status === "Converged" ? "ok" : status === "Mild Drift" ? "warn" : "bad";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.filename}</td>
        <td><strong>${formatLabelHtml(r.name)}</strong></td>
        <td>${r.count}</td>
        <td>${fmt(r.min)}</td>
        <td>${fmt(r.max)}</td>
        <td>${fmt(r.mean)} ± ${fmt(r.std)}</td>
        <td>${fmt(r.median)}</td>
        <td>${fmt(r.eq_mean)} ± ${fmt(r.eq_std)}</td>
        <td>${fmt(r.block_sem)}</td>
        <td>${Number.isFinite(r.n_eff) ? Math.round(r.n_eff) : "—"}</td>
        <td><span class="conv-pill ${cls}">${status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function applyZoom() {
    const sheet = $("paper-sheet");
    sheet.classList.toggle("zoom-fit", state.zoom === "fit");
    if (state.zoom === "fit") {
      sheet.style.transform = "";
    } else {
      sheet.style.transform = `scale(${state.zoom})`;
      sheet.style.transformOrigin = "top center";
    }
  }

  async function exportFigure(fmt) {
    setRenderStatus("busy", `Exporting ${fmt.toUpperCase()}…`);
    try {
      const blob = await backend.renderBlob(
        state.spec,
        fmt,
        state.spec.dpi,
        `gmxgrace_${state.spec.journal}`
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `gmxgrace_${state.spec.journal}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setRenderStatus("ok", "Live SVG Ready");
      showToast(`Downloaded publication ${fmt.toUpperCase()}`);
    } catch (err) {
      setRenderStatus("error", "Export failed");
      showToast(`Export error: ${err.message}`);
    }
  }

  // -------------------------------------------------------------------------
  // Event Binding
  // -------------------------------------------------------------------------

  function bindStaticEvents() {
    // Dropzone & file input
    const dz = $("dropzone");
    const fi = $("file-input");
    dz.addEventListener("click", () => fi.click());
    fi.addEventListener("change", (e) => {
      handleFiles(e.target.files);
      fi.value = "";
    });
    dz.addEventListener("dragover", (e) => {
      e.preventDefault();
      dz.classList.add("drag-over");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("drag-over");
      handleFiles(e.dataTransfer.files);
    });

    // Floating Inline SVG Label Editor events
    if ($("inline-editor")) {
      $("inline-editor-save")?.addEventListener("click", () => activeInlineCommit && activeInlineCommit());
      $("inline-editor-cancel")?.addEventListener("click", () => closeInlineEditor());
      $("inline-editor-input")?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          if (activeInlineCommit) activeInlineCommit();
        } else if (e.key === "Escape") {
          e.preventDefault();
          closeInlineEditor();
        }
      });
      document.addEventListener("mousedown", (e) => {
        const box = $("inline-editor");
        if (!box || box.classList.contains("hidden")) return;
        if (!box.contains(e.target) && !(e.target.closest && e.target.closest('[id^="gmx_edit__"]'))) {
          closeInlineEditor();
        }
      });
    }

    // Academic License Modal
    const licModal = $("license-modal");
    if ($("btn-license") && licModal) {
      const closeModal = () => licModal.classList.add("hidden");
      $("btn-license").addEventListener("click", () => licModal.classList.remove("hidden"));
      $("btn-close-license")?.addEventListener("click", closeModal);
      $("btn-accept-license")?.addEventListener("click", closeModal);
      licModal.addEventListener("click", (e) => {
        if (e.target === licModal) closeModal();
      });
    }

    // Export buttons
    document.querySelectorAll("[data-export]").forEach((btn) => {
      btn.addEventListener("click", () => exportFigure(btn.dataset.export));
    });
    $("btn-export-recipe").addEventListener("click", () => {
      if (!state.lastRecipe) return;
      const blob = new Blob([JSON.stringify(state.lastRecipe, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "figure_recipe.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      showToast("Downloaded reproducible figure_recipe.json");
    });

    // Import saved recipe JSON
    const recipeInput = $("recipe-input");
    $("btn-import-recipe").addEventListener("click", () => recipeInput.click());
    recipeInput.addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      recipeInput.value = "";
      if (!f) return;
      try {
        const parsed = JSON.parse(await f.text());
        const incomingSpec = parsed.spec || parsed;
        if (!incomingSpec.panels || !Array.isArray(incomingSpec.panels)) {
          throw new Error("Invalid recipe JSON (missing panels)");
        }
        // Remap dataset IDs by matching filenames if already loaded
        const nameToId = {};
        Object.entries(state.datasets).forEach(([id, ds]) => {
          nameToId[ds.filename] = id;
        });
        const filesMap = parsed.files || {};
        incomingSpec.panels.forEach((p) => {
          p.dataset_ids = (p.dataset_ids || []).map((oldId) => {
            if (state.datasets[oldId]) return oldId;
            const fname = filesMap[oldId];
            return (fname && nameToId[fname]) ? nameToId[fname] : oldId;
          });
        });
        state.spec = incomingSpec;
        state.activePanelIdx = 0;
        syncAllControls();
        await renderNow();
        showToast("Loaded figure recipe!");
      } catch (err) {
        showToast(`Recipe load error: ${err.message}`);
      }
    });

    // Export Statistics CSV
    $("btn-export-stats-csv").addEventListener("click", () => {
      if (!state.lastStats.length) {
        showToast("No XY statistics to export");
        return;
      }
      const header = ["filename", "series", "count", "min", "max", "mean", "std", "median", "eq_mean", "eq_std", "block_sem", "drift_total", "tau_frames", "n_eff", "convergence_status"];
      const lines = [header.join(",")];
      state.lastStats.forEach((r) => {
        const vals = [r.filename, r.name, r.count, r.min, r.max, r.mean, r.std, r.median, r.eq_mean, r.eq_std, r.block_sem, r.drift_total, r.tau_frames, r.n_eff, r.convergence_status]
          .map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`);
        lines.push(vals.join(","));
      });
      const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "gmxgrace_statistics.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      showToast("Exported gmxgrace_statistics.csv");
    });

    // Layout grid buttons
    document.querySelectorAll(".layout-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const [r, c] = btn.dataset.grid.split("x").map(Number);
        state.spec.nrows = r;
        state.spec.ncols = c;
        const needed = r * c;
        while (state.spec.panels.length < needed) {
          state.spec.panels.push(createDefaultPanel(`panel-${state.spec.panels.length + 1}`));
        }
        state.spec.panels = state.spec.panels.slice(0, needed);
        if (state.activePanelIdx >= needed) state.activePanelIdx = 0;
        if (c >= 2 && state.spec.column_width === "single") {
          state.spec.column_width = "double";
        }
        syncAllControls();
        scheduleRender();
      });
    });

    $("panel-labels-select").addEventListener("change", (e) => {
      state.spec.panel_labels = e.target.value;
      scheduleRender();
    });

    // Zoom & refresh
    document.querySelectorAll("[data-zoom]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-zoom]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.zoom = btn.dataset.zoom === "fit" ? "fit" : Number(btn.dataset.zoom);
        applyZoom();
      });
    });
    $("btn-refresh").addEventListener("click", () => renderNow());

    // Bottom drawer tabs & collapse
    document.querySelectorAll(".drawer-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".drawer-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        $("drawer-reproduce").classList.toggle("hidden", tab.dataset.drawer !== "reproduce");
        $("drawer-stats").classList.toggle("hidden", tab.dataset.drawer !== "stats");
        if ($("drawer-fel")) {
          $("drawer-fel").classList.toggle("hidden", tab.dataset.drawer !== "fel");
        }
        $("bottom-drawer").classList.remove("collapsed");
      });
    });
    $("btn-toggle-drawer").addEventListener("click", () => {
      const dr = $("bottom-drawer");
      dr.classList.toggle("collapsed");
      $("btn-toggle-drawer").textContent = dr.classList.contains("collapsed") ? "▴" : "▾";
    });

    // Copy buttons
    document.querySelectorAll("[data-copy]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = $(btn.dataset.copy).textContent;
        try {
          await navigator.clipboard.writeText(text);
          showToast("Copied to clipboard!");
        } catch {
          showToast("Copy blocked by browser permissions");
        }
      });
    });

    // Right inspector tabs
    document.querySelectorAll(".insp-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".insp-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        ["style", "panel", "analysis", "series"].forEach((name) => {
          $(`tab-${name}`).classList.toggle("hidden", tab.dataset.tab !== name);
        });
      });
    });

    // Global style controls
    $("ctrl-width").addEventListener("change", (e) => {
      state.spec.column_width = e.target.value;
      $("custom-dims-row").classList.toggle("hidden", e.target.value !== "custom");
      scheduleRender();
    });
    $("ctrl-width-mm").addEventListener("input", (e) => {
      state.spec.width_mm = parseOptionalFloat(e.target.value);
      scheduleRender();
    });
    $("ctrl-height-mm").addEventListener("input", (e) => {
      state.spec.height_mm = parseOptionalFloat(e.target.value);
      scheduleRender();
    });
    $("ctrl-dpi").addEventListener("change", (e) => {
      state.spec.dpi = Number(e.target.value);
      updateDimensionsReadout();
    });
    $("ctrl-aspect").addEventListener("input", (e) => {
      state.spec.aspect_ratio = Number(e.target.value);
      $("val-aspect").textContent = state.spec.aspect_ratio.toFixed(2);
      scheduleRender();
    });
    $("ctrl-font-family").addEventListener("change", (e) => {
      state.spec.font_family = e.target.value || null;
      scheduleRender();
    });
    $("ctrl-font-size").addEventListener("input", (e) => {
      state.spec.font_size = parseOptionalFloat(e.target.value);
      scheduleRender();
    });
    $("ctrl-line-width").addEventListener("input", (e) => {
      state.spec.line_width = parseOptionalFloat(e.target.value);
      scheduleRender();
    });
    $("ctrl-transparent").addEventListener("change", (e) => {
      state.spec.transparent = e.target.checked;
      $("paper-sheet")?.classList.toggle("transparent-bg", e.target.checked);
      scheduleRender();
    });
    $("ctrl-suptitle").addEventListener("input", (e) => {
      state.spec.suptitle = e.target.value;
      scheduleRender();
    });
    if ($("ctrl-max-pts")) {
      $("ctrl-max-pts").addEventListener("change", (e) => {
        state.spec.max_plot_points = Number(e.target.value);
        scheduleRender();
      });
    }

    // Panel Axes controls
    const bindPanel = (id, key, transform = (v) => v, evt = "input") => {
      $(id).addEventListener(evt, (e) => {
        const val = e.target.type === "checkbox" ? e.target.checked : e.target.value;
        getActivePanel()[key] = transform(val);
        scheduleRender();
      });
    };

    bindPanel("p-title", "title");
    bindPanel("p-plot-type", "plot_type", (v) => v, "change");
    bindPanel("p-legend-loc", "legend_loc", (v) => v, "change");
    bindPanel("p-x-unit", "x_unit", (v) => v, "change");
    bindPanel("p-y-unit", "y_unit", (v) => v, "change");
    bindPanel("p-xlabel", "x_label", (v) => (v.trim() === "" ? null : v));
    bindPanel("p-ylabel", "y_label", (v) => (v.trim() === "" ? null : v));
    if ($("p-y2label")) bindPanel("p-y2label", "y2_label", (v) => (v.trim() === "" ? null : v));
    if ($("p-y2-unit")) bindPanel("p-y2-unit", "y2_unit", (v) => v, "change");
    bindPanel("p-xmin", "x_min", parseOptionalFloat);
    bindPanel("p-xmax", "x_max", parseOptionalFloat);
    bindPanel("p-ymin", "y_min", parseOptionalFloat);
    bindPanel("p-ymax", "y_max", parseOptionalFloat);
    bindPanel("p-xscale", "x_scale", (v) => v, "change");
    bindPanel("p-yscale", "y_scale", (v) => v, "change");
    bindPanel("p-show-legend", "show_legend", Boolean, "change");
    bindPanel("p-legend-frame", "legend_frame", Boolean, "change");
    bindPanel("p-grid", "grid", Boolean, "change");
    bindPanel("p-minor-ticks", "minor_ticks", Boolean, "change");

    // Analysis controls
    bindPanel("p-aggregate", "aggregate", (v) => v, "change");
    bindPanel("p-agg-label", "aggregate_label");
    bindPanel("p-agg-indiv", "aggregate_show_individual", Boolean, "change");
    bindPanel("p-smooth", "smoothing_method", (v) => v, "change");
    $("p-window").addEventListener("input", (e) => {
      const w = Number(e.target.value);
      getActivePanel().smoothing_window = w;
      $("val-window").textContent = String(w);
      scheduleRender();
    });
    bindPanel("p-show-raw", "show_raw_under_smooth", Boolean, "change");
    bindPanel("p-marginal", "marginal_hist", Boolean, "change");
    bindPanel("p-mean-line", "show_mean_line", Boolean, "change");
    $("p-eq").addEventListener("input", (e) => {
      const f = Number(e.target.value);
      getActivePanel().equilibration_fraction = f;
      const pct = `${Math.round(f * 100)}%`;
      $("val-eq").textContent = pct;
      $("stats-eq-label").textContent = pct;
      scheduleRender();
    });

    // Matrix controls
    bindPanel("p-cmap", "cmap", (v) => v, "change");
    bindPanel("p-contours", "contour_levels", (v) => Math.max(2, Number(v) || 10));
    bindPanel("p-cbar-label", "colorbar_label", (v) => (v.trim() === "" ? null : v));
    bindPanel("p-show-contours", "show_contours", Boolean, "change");

    // FEL 2D, System Comparison & Convergence controls
    if ($("p-fel-temp")) bindPanel("p-fel-temp", "fel_temp_k", (v) => Math.max(10, Number(v) || 300));
    if ($("p-fel-bins")) bindPanel("p-fel-bins", "fel_bins", (v) => Math.max(10, Number(v) || 45));
    if ($("p-fel-unit")) bindPanel("p-fel-unit", "fel_unit", (v) => v, "change");
    if ($("p-fel-minima")) bindPanel("p-fel-minima", "fel_mark_minima", Boolean, "change");
    if ($("p-stat-test")) bindPanel("p-stat-test", "stat_test", (v) => v, "change");
    if ($("p-conv-blocks")) bindPanel("p-conv-blocks", "convergence_blocks", (v) => Math.max(2, Number(v) || 5));
    if ($("p-show-pvalues")) bindPanel("p-show-pvalues", "show_pvalues", Boolean, "change");

    // Zoom Inset controls
    const ensureInset = () => {
      const p = getActivePanel();
      if (!p.zoom_inset) {
        p.zoom_inset = { enabled: false, x_min: null, x_max: null, y_min: null, y_max: null, loc: "lower right", width_pct: 36, height_pct: 34 };
      }
      return p.zoom_inset;
    };
    if ($("p-inset-enabled")) {
      $("p-inset-enabled").addEventListener("change", (e) => {
        ensureInset().enabled = e.target.checked;
        scheduleRender();
      });
      $("p-inset-xmin").addEventListener("input", (e) => {
        ensureInset().x_min = parseOptionalFloat(e.target.value);
        scheduleRender();
      });
      $("p-inset-xmax").addEventListener("input", (e) => {
        ensureInset().x_max = parseOptionalFloat(e.target.value);
        scheduleRender();
      });
      $("p-inset-loc").addEventListener("change", (e) => {
        ensureInset().loc = e.target.value;
        scheduleRender();
      });
    }

    // Protein Domain Track builder
    if ($("btn-add-dom")) {
      $("btn-add-dom").addEventListener("click", () => {
        const start = parseOptionalFloat($("dom-start").value);
        const end = parseOptionalFloat($("dom-end").value);
        const label = $("dom-label").value.trim();
        const color = $("dom-color").value || "#3B82F6";
        if (start === null || end === null || end <= start || !label) {
          showToast("Enter start < end residue numbers and a domain label");
          return;
        }
        const p = getActivePanel();
        if (!p.domain_tracks) p.domain_tracks = [];
        p.domain_tracks.push({ start, end, label, color });
        $("dom-start").value = "";
        $("dom-end").value = "";
        $("dom-label").value = "";
        renderDomainsList();
        scheduleRender();
      });
    }

    // Annotations
    $("btn-add-ann").addEventListener("click", () => {
      const val = parseOptionalFloat($("ann-val").value);
      if (val === null) {
        showToast("Enter a numeric coordinate for the annotation");
        return;
      }
      const kind = $("ann-kind").value;
      const valEnd = parseOptionalFloat($("ann-val-end").value);
      if ((kind === "vspan" || kind === "hspan") && valEnd === null) {
        showToast("Enter an End coordinate for a shaded region");
        return;
      }
      getActivePanel().annotations.push({
        kind,
        value: val,
        value_end: valEnd,
        text: $("ann-text").value.trim(),
        color: "#475569",
        linestyle: "--",
        linewidth: 0.9,
        alpha: 0.75,
      });
      $("ann-val").value = "";
      $("ann-val-end").value = "";
      $("ann-text").value = "";
      renderAnnotationsList();
      scheduleRender();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
