/* GMXgrace Pyodide Web Worker — runs the exact same Python/Matplotlib engine locally in browser WASM */

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/";

let pyodide = null;
let readyPromise = null;
let edrInstalled = false;

function reportProgress(stage, pct) {
  self.postMessage({ type: "progress", stage, pct });
}

async function initEngine(wheelUrl) {
  reportProgress("Downloading Python WebAssembly runtime (cached after first load)…", 15);
  importScripts(`${PYODIDE_INDEX}pyodide.js`);

  pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

  reportProgress("Loading NumPy, SciPy, Matplotlib, Pillow & Pydantic…", 45);
  await pyodide.loadPackage(["numpy", "scipy", "matplotlib", "pillow", "pydantic"]);

  reportProgress("Installing GMXgrace publication engine…", 78);
  const resolvedWheel = new URL(wheelUrl, self.location.href).href;
  const wheelResp = await fetch(resolvedWheel);
  if (!wheelResp.ok) {
    throw new Error(`Failed to fetch GMXgrace wheel (${wheelResp.status}): ${resolvedWheel}`);
  }
  const wheelBuf = await wheelResp.arrayBuffer();
  pyodide.unpackArchive(wheelBuf, "zip", { extractDir: "/home/pyodide/gmxgrace_pkg" });

  reportProgress("Initializing Matplotlib publication backend…", 92);
  await pyodide.runPythonAsync(`
import json
import sys
if "/home/pyodide/gmxgrace_pkg" not in sys.path:
    sys.path.insert(0, "/home/pyodide/gmxgrace_pkg")
import matplotlib
matplotlib.use("Agg")
from gmxgrace.session import StudioSession

_session = StudioSession()

def _rpc_presets():
    return json.dumps(_session.get_presets())

def _rpc_sample(bundle_id):
    return json.dumps(_session.load_sample_bundle(bundle_id))

def _rpc_delete(ds_id):
    return json.dumps({"deleted": _session.delete_dataset(ds_id)})

def _rpc_upload(file_items, nxy):
    pairs = []
    for item in file_items:
        name = str(item["name"])
        raw = bytes(item["bytes"].to_py())
        pairs.append((name, raw))
    return json.dumps(_session.upload_files(pairs, nxy=bool(nxy)))

def _rpc_render(spec_json, fmt, dpi):
    spec = json.loads(spec_json)
    return _session.render(spec, fmt=fmt, dpi=(int(dpi) if dpi else None))

def _rpc_codegen(spec_json, output_name):
    spec = json.loads(spec_json)
    return json.dumps(_session.codegen(spec, output_name=output_name))

def _rpc_stats(dataset_ids_json, eq_fraction):
    ds_ids = json.loads(dataset_ids_json)
    return json.dumps(_session.stats(ds_ids, eq_fraction=float(eq_fraction)))
`);

  reportProgress("Browser Python engine ready!", 100);
}

async function ensureEdrSupport() {
  if (edrInstalled) return;
  reportProgress("Loading pyedr for binary GROMACS .edr file…", 60);
  try {
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install(["mda-xdrlib", "pyedr"])
`);
    edrInstalled = true;
  } catch (err) {
    console.warn("Could not install pyedr in browser:", err);
  }
}

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  try {
    if (action === "init") {
      if (!readyPromise) {
        readyPromise = initEngine(payload.wheelUrl);
      }
      await readyPromise;
      self.postMessage({ id, ok: true, data: { ready: true } });
      return;
    }

    await readyPromise;

    if (action === "presets") {
      const jsonStr = pyodide.globals.get("_rpc_presets")();
      self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      return;
    }

    if (action === "sample") {
      const jsonStr = pyodide.globals.get("_rpc_sample")(payload.bundleId);
      self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      return;
    }

    if (action === "delete") {
      const jsonStr = pyodide.globals.get("_rpc_delete")(payload.datasetId);
      self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      return;
    }

    if (action === "upload") {
      const hasEdr = payload.files.some((f) => f.name.toLowerCase().endsWith(".edr"));
      if (hasEdr) {
        await ensureEdrSupport();
      }
      const pyItems = pyodide.toPy(payload.files);
      try {
        const jsonStr = pyodide.globals.get("_rpc_upload")(pyItems, payload.nxy);
        self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      } finally {
        pyItems.destroy();
      }
      return;
    }

    if (action === "render") {
      const pyBytes = pyodide.globals.get("_rpc_render")(
        JSON.stringify(payload.spec),
        payload.fmt || "svg",
        payload.dpi || null
      );
      try {
        const u8 = pyBytes.toJs({ create_proxies: false });
        // Transfer underlying ArrayBuffer with zero copy for crisp speed
        const copy = new Uint8Array(u8);
        self.postMessage({ id, ok: true, data: copy.buffer }, [copy.buffer]);
      } finally {
        pyBytes.destroy();
      }
      return;
    }

    if (action === "codegen") {
      const jsonStr = pyodide.globals.get("_rpc_codegen")(
        JSON.stringify(payload.spec),
        payload.outputName || "figure.pdf"
      );
      self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      return;
    }

    if (action === "stats") {
      const jsonStr = pyodide.globals.get("_rpc_stats")(
        JSON.stringify(payload.datasetIds || []),
        payload.eqFraction || 0.0
      );
      self.postMessage({ id, ok: true, data: JSON.parse(jsonStr) });
      return;
    }

    throw new Error(`Unknown worker action: ${action}`);
  } catch (err) {
    self.postMessage({ id, ok: false, error: err?.message || String(err) });
  }
};
