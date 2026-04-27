"""Dataclasses shared by every reconstruction back-end.

The orchestrator in ``spectra/pipeline.py`` uses a single `RawReconstruction`
shape so it can apply ArUco-based Sim3 alignment, run surface
reconstruction, and write outputs in a backend-agnostic way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np


@dataclass
class BackendInputs:
    """Resolved inputs passed to a backend's `run_*` function."""

    image_paths: List[Path]
    pose_dir: Path | None = None
    camera_params_dir: Path | None = None
    # Optional pre-loaded tensors (the loader helper below fills these in):
    T_world_cam_gt: np.ndarray | None = None     # [V, 4, 4], camera-to-world
    K_orig_gt: np.ndarray | None = None          # [3, 3], shared intrinsics
    cam2ee: np.ndarray | None = None             # [4, 4], camera-in-EE
    distortion_coeffs: np.ndarray | None = None  # [5], distortion (unused today)


@dataclass
class RawReconstruction:
    """Output of a back-end, expressed in a consistent "world" frame.

    - Clouds and point maps live in ``T_world_cam``'s frame.
    - ``K_per_view_orig`` is intrinsics expressed in the *original* image
      pixel space — the ArUco alignment stage needs this because 2D marker
      detections are computed on full-resolution images.
    """

    fused_points: np.ndarray              # [N, 3] float32
    fused_colors: np.ndarray              # [N, 3] uint8
    fused_confidence: np.ndarray          # [N]   float32
    point_map_world: np.ndarray           # [V, Hn, Wn, 3] float32
    valid_masks: np.ndarray               # [V, Hn, Wn] bool
    T_world_cam: np.ndarray               # [V, 4, 4] float32
    K_per_view_orig: np.ndarray           # [V, 3, 3] float32 (original pixels)
    K_per_view_network: np.ndarray        # [V, 3, 3] float32 (network pixels)
    network_image_sizes: np.ndarray       # [V, 2] int32, (width, height)
    original_image_sizes: np.ndarray      # [V, 2] int32, (width, height)
    images_network_uint8: np.ndarray      # [V, Hn, Wn, 3] uint8
    confidence_maps_network: np.ndarray   # [V, Hn, Wn] float32
    frame_description: str                # 'predicted' | 'gt' | 'gt-aligned'
    backend_name: str                     # 'mast3r'
    alignment_info: dict | None = None    # backend-specific pred->GT Sim3 info
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["BackendInputs", "RawReconstruction"]
