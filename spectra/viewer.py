"""Local Gradio 3D viewer for reconstruction runs.

Launches a web app at http://127.0.0.1:7860 that lets you pick any run from the
output folder (default ``RESULTS/``) and explore its fused point cloud +
triangulated ArUco markers in 3D, optionally overlaid on the surface mesh.

The viewer assembles a single GLB per (run, max_points, include_surface) on
demand and hands it to ``gradio.Model3D``:

- **Fused cloud**: colored ``trimesh.PointCloud`` (alpha 255), optionally
  downsampled to the slider value for browser-side rendering speed.
- **ArUcos**: one ``trimesh.Trimesh`` per triangulated marker (two triangles
  forming the quad, per-ID stable color, alpha 230 for semi-transparency).
- **Surface** (optional toggle): ``trimesh.Trimesh`` loaded from
  ``<run>/surface_mesh.ply`` when present.

Usage::

    spectra viewer                                  # browse ./RESULTS/
    spectra viewer --results-dir RESULTS --port 7861
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import trimesh

from .aruco import color_for_id_rgb


DEFAULT_RESULTS_DIR = Path("RESULTS")
DEFAULT_MAX_POINTS = 150_000
GLB_CACHE_NAME = "viewer.glb"


def _list_runs(results_dir: Path) -> List[str]:
    """Return run subfolders (newest first) that contain a fused cloud."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    runs = [
        p
        for p in results_dir.iterdir()
        if p.is_dir() and (p / "cloud.npz").exists() and not p.is_symlink()
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def _subsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample the cloud when it exceeds ``max_points``."""
    n = int(points.shape[0])
    if max_points <= 0 or n <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return points[idx], colors[idx]


def _apply_sim3(points: np.ndarray, sim3: Optional[np.ndarray]) -> np.ndarray:
    """Apply a 4x4 similarity transform to a batch of points (identity if None)."""
    if sim3 is None:
        return points
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)  # (N, 3)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)  # (N, 4)
    out = homo @ np.asarray(sim3, dtype=np.float32).T             # (N, 4)
    return out[:, :3]


def _build_marker_mesh(corners_3d: np.ndarray, marker_id: int) -> trimesh.Trimesh:
    """Build a two-triangle quad mesh with a per-ID color for one ArUco."""
    r, g, b = color_for_id_rgb(int(marker_id))
    rgba = np.asarray([r, g, b, 230], dtype=np.uint8)
    vertex_colors = np.tile(rgba, (4, 1))                                 # (4, 4)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)        # (2, 3)
    return trimesh.Trimesh(
        vertices=np.asarray(corners_3d, dtype=np.float32).reshape(4, 3),
        faces=triangles,
        vertex_colors=vertex_colors,
        process=False,
    )


def _build_glb(
    run_dir: Path,
    max_points: int = DEFAULT_MAX_POINTS,
    include_surface: bool = False,
) -> Path:
    """Assemble a GLB combining the cloud + ArUco quads (+ optional surface)."""
    run_dir = Path(run_dir)
    cloud_npz = np.load(run_dir / "cloud.npz")
    points = cloud_npz["points"].astype(np.float32)       # (N, 3)
    colors = cloud_npz["colors"].astype(np.uint8)         # (N, 3) uint8 RGB
    points, colors = _subsample(points, colors, max_points)

    scene = trimesh.Scene()

    rgba = np.concatenate(
        [colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1
    )                                                     # (N, 4) uint8 RGBA
    cloud_geom = trimesh.PointCloud(vertices=points, colors=rgba)
    scene.add_geometry(cloud_geom, geom_name="cloud")

    aruco_json = run_dir / "aruco_markers_3d.json"
    sim3_path = run_dir / "sim3_to_output_frame.npy"
    sim3 = np.load(sim3_path) if sim3_path.exists() else None   # (4, 4) maps input→output
    if aruco_json.exists():
        data = json.loads(aruco_json.read_text(encoding="utf-8"))
        for mid_str, marker in data.get("markers", {}).items():
            corners_input = np.asarray(marker["corners_3d"], dtype=np.float32)   # (4, 3)
            corners_output = _apply_sim3(corners_input, sim3)                    # (4, 3)
            mesh = _build_marker_mesh(
                corners_3d=corners_output,
                marker_id=int(mid_str),
            )
            scene.add_geometry(mesh, geom_name=f"aruco_{mid_str}")

    if include_surface:
        surface_path = run_dir / "surface_mesh.ply"
        if surface_path.exists():
            surface_mesh = trimesh.load(surface_path, process=False, force="mesh")
            if isinstance(surface_mesh, trimesh.Trimesh):
                scene.add_geometry(surface_mesh, geom_name="surface")

    out_path = run_dir / GLB_CACHE_NAME
    scene.export(out_path)
    return out_path


def _load_viewer(
    run_name: Optional[str],
    max_points: float,
    include_surface: bool,
    results_dir: Path,
) -> Optional[str]:
    """Gradio callback: build a GLB for the selected run and return its path."""
    if not run_name:
        return None
    run_dir = Path(results_dir) / run_name
    if not run_dir.is_dir():
        return None
    try:
        glb_path = _build_glb(
            run_dir=run_dir,
            max_points=int(max_points),
            include_surface=bool(include_surface),
        )
    except Exception as exc:  # surface rendering in the Gradio log
        raise gr.Error(f"Failed to build GLB for {run_name!r}: {exc}") from exc
    return str(glb_path)


def _run_summary(run_name: Optional[str], results_dir: Path) -> str:
    """Return a short Markdown summary of the selected run (counts + scale)."""
    if not run_name:
        return "_No run selected._"
    run_dir = Path(results_dir) / run_name
    meta_path = run_dir / "reconstruction_metadata.json"
    if not meta_path.exists():
        return f"**{run_name}** — no `reconstruction_metadata.json` found."
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = [
        f"**{run_name}**",
        f"- backend: `{meta.get('backend', '?')}`",
        f"- cloud points: `{meta.get('num_cloud_points', 0):,}`",
        f"- surface vertices: `{meta.get('num_surface_points', 0):,}`",
        f"- surface triangles: `{meta.get('num_surface_triangles', 0):,}`",
        f"- ArUco detections (2D): `{meta.get('num_aruco_detections', 0)}`",
    ]
    align = meta.get("aruco_alignment")
    if align:
        lines.append(
            f"- ArUco markers (3D): `{align.get('num_markers', 0)}` "
            f"[ids: {align.get('used_marker_ids', [])}]"
        )
        lines.append(
            f"- metric scale: `{align.get('scale_input_to_meters', 1.0):.6f}` "
            f"(MAD `{align.get('scale_mad', 0.0):.6f}`, "
            f"applied `{align.get('sim3_scale_applied', 1.0):.6f}`)"
        )
    return "\n".join(lines)


def _build_ui(results_dir: Path) -> gr.Blocks:
    initial_runs = _list_runs(results_dir)
    initial_run = initial_runs[0] if initial_runs else None

    with gr.Blocks(title="Spectra 3D viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Spectra 3D viewer")
        gr.Markdown(
            f"Browsing runs under `{results_dir}`. Filled colored squares are "
            "triangulated ArUcos (per-ID deterministic color). Drag to orbit, "
            "scroll to zoom."
        )

        with gr.Row():
            run_dd = gr.Dropdown(
                label="Run",
                choices=initial_runs,
                value=initial_run,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh list", variant="secondary")

        with gr.Row():
            max_points_slider = gr.Slider(
                minimum=10_000,
                maximum=1_000_000,
                value=DEFAULT_MAX_POINTS,
                step=10_000,
                label="Max points (downsample for browser speed)",
            )
            surface_chk = gr.Checkbox(value=False, label="Overlay surface mesh")

        viewer = gr.Model3D(
            label="Point cloud + ArUcos",
            interactive=False,
            height=720,
            clear_color=[0.05, 0.05, 0.07, 1.0],
        )

        summary = gr.Markdown(_run_summary(initial_run, results_dir))

        def _on_change(run_name, max_pts, surface):
            glb = _load_viewer(run_name, max_pts, surface, results_dir)
            return glb, _run_summary(run_name, results_dir)

        for component in (run_dd, max_points_slider, surface_chk):
            component.change(
                _on_change,
                inputs=[run_dd, max_points_slider, surface_chk],
                outputs=[viewer, summary],
            )

        def _on_refresh():
            runs = _list_runs(results_dir)
            value = runs[0] if runs else None
            return gr.Dropdown(choices=runs, value=value)

        refresh_btn.click(_on_refresh, outputs=run_dd)

        if initial_run is not None:
            demo.load(
                _on_change,
                inputs=[run_dd, max_points_slider, surface_chk],
                outputs=[viewer, summary],
            )

    return demo


def run_viewer(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    """Launch the Gradio viewer app on ``http://{host}:{port}``."""
    results_dir = Path(results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    demo = _build_ui(results_dir)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )


__all__ = ["run_viewer"]
