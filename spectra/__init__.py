"""Unified 3D reconstruction pipeline with ArUco-stabilized alignment.

Public API highlights:

- `run_reconstruction(cfg)` — top-level entry point that dispatches to a
  back-end (VGGT or MASt3R), stabilizes the output on the ArUco plane, and
  writes cloud + surface + normals to a timestamped run folder.
- `ReconstructionConfig` — pydantic-backed YAML config schema.
- `ArucoDetector`, `detect_image`, `detect_folder` — ArUco utilities.
"""

from __future__ import annotations

from .aruco import (
    ARUCO_DICTIONARIES,
    ArucoDetector,
    MarkerDetection,
    annotate_image,
    color_for_id,
    color_for_id_rgb,
    detect_folder,
    detect_image,
    read_detections_json,
)
from .config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    ReconstructionConfig,
    SurfaceConfig,
    VggtConfig,
    load_config,
    save_config,
)
from .pipeline import ReconstructionResult, run_reconstruction


def run_viewer(*args, **kwargs):
    """Lazy re-export of :func:`spectra.viewer.run_viewer` (needs ``gradio``)."""
    from .viewer import run_viewer as _run_viewer

    return _run_viewer(*args, **kwargs)

__all__ = [
    "ARUCO_DICTIONARIES",
    "ArucoConfig",
    "ArucoDetector",
    "InputConfig",
    "MarkerDetection",
    "Mast3rConfig",
    "OutputConfig",
    "ReconstructionConfig",
    "ReconstructionResult",
    "SurfaceConfig",
    "VggtConfig",
    "annotate_image",
    "color_for_id",
    "color_for_id_rgb",
    "detect_folder",
    "detect_image",
    "load_config",
    "read_detections_json",
    "run_reconstruction",
    "run_viewer",
    "save_config",
]
