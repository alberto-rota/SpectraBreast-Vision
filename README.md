# SpectraBreast — Vision Pipeline 
Contributors: **Alberto Rota**, **Leonardo Passoni**, **Anna Bicchi**

## Installation
Use with [uv](https://docs.astral.sh/uv/) as your environemnt manager. See https://docs.astral.sh/uv/#installation for installation guidelines.

Clone-to-run:

```bash
git clone https://github.com/alberto-rota/SpectraBreast-Vision
cd SpectraBreast-Vision
uv sync
uv run spectra --help
```

This project is now fully managed by `uv`:

- Dependencies are declared in `pyproject.toml` (including local editable `mast3r/asmk` via `tool.uv.sources`).
- The CLI entrypoint is declared in `project.scripts`.
- You can run commands without manual activation via `uv run ...`.

If you prefer activating a shell (won't require `uv run` before every command):

```bash
source .venv/bin/activate
spectra --help
```

## CLI commands

- `spectra recon` - Run the 3D reconstruction pipeline 
- `spectra detect` - 2D ArUco detection
- `spectra viewer` - Gradio browser to inspect outputs under `RESULTS/`
- `spectra calibrate-intrinsics` - checkerboard-based camera intrinsic calibration

```bash
# Config-driven
uv run spectra recon --config configs/default.yaml

# Overrides
uv run spectra recon -c configs/default.yaml --rgb-dir DATA/s/rgb --out RESULTS --run-name trial1

# [WORK IN PROGRESS] Optional camera pose and intrinsic GT: same stem order as images → pose_*.txt + intrinsics.npy folder
uv run spectra recon -c configs/default.yaml \
  --rgb-dir DATA/s/rgb --pose-dir DATA/s/poses --camera-params-dir DATA/s/cam

# 2D detection only
uv run spectra detect DATA/s/rgb /tmp/aruco_out

# Intrinsic calibration from checkerboard images
uv run spectra calibrate-intrinsics --image-dir checkerboard --output-dir intrinsics

# Arbitrary YAML field
uv run spectra recon -c configs/default.yaml -s mast3r.voxel_size=0.002
```


Notes:

- `uv run spectra`, `uv run python -m spectra`, and `uv run python -m spectra.cli` expose the same command tree (`run`, `detect`, `viewer`, hidden aliases).
- `spectra viewer` / `spectra tui` are the CLI wrappers around `spectra.viewer.run_viewer` and `spectra.tui.run_tui`.
- `spectra calibrate-intrinsics` wraps `spectra.calibration.calibrate_intrinsics`.
- `mast3r_pipeline.py` is backward compatibility only; prefer `spectra recon`.

## Input layout

```
DATA/myset/
  rgb/              # required: *.jpg / *.png
  poses/            # optional: pose_0000.txt … (one line: X Y Z R P Y)
  camera_parameters/  # optional: intrinsics.npy (3×3), camera2ee.npy (ignored by fusion)
```

Relative paths in YAML are resolved from the current working directory.

## Outputs (per run folder under `output.root` / `run_name` or timestamp)

- `cloud.ply`, `cloud.npz` — fused RGB point cloud  
- `aruco_detections/json/`, `annotated/` — 2D ArUco  
- `aruco_markers_3d.json` — 3D marker corners in the output frame (when alignment succeeds)  
- `reconstruction_metadata.json` — run summary  
- `run.yaml` / `run.json` — resolved configuration  
- `rerun/spectra.rrd` — optional Rerun recording (if enabled)
