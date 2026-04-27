"""Typed configuration schema for the ArUco-stabilized reconstruction pipeline.

Configuration is stored as YAML on disk and round-tripped through pydantic v2
models so that CLI overrides, TUI edits, and API callers all see a single
validated object.

Typical usage::

    from spectra.config import load_config, ReconstructionConfig

    cfg = load_config("configs/default.yaml")
    cfg = cfg.with_overrides({"aruco.marker_edge_length_m": 0.03})

The pipeline is **MASt3R-SfM** only (images → dense cloud + ArUco 2D/3D). Optional
``pose_dir`` and ``camera_params_dir`` are supported when available.

CLI overrides use dotted paths, e.g. ``--set aruco.marker_edge_length_m=0.03``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .aruco import ARUCO_DICTIONARIES


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rgb_dir: Path = Field(..., description="Folder containing input images (jpg/png).")
    pose_dir: Path | None = Field(
        default=None,
        description=(
            "Optional folder with pose_*.txt files (one per image, 6 floats "
            "[X Y Z Roll Pitch Yaw]). When None, runs in RGB-only mode."
        ),
    )
    camera_params_dir: Path | None = Field(
        default=None,
        description=(
            "Optional folder with intrinsics.npy and camera2ee.npy. When "
            "None, the back-end estimates intrinsics."
        ),
    )


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Field(default=Path("RESULTS"), description="Root folder for all runs.")
    run_name: str | None = Field(
        default=None,
        description="Optional subfolder name. If None, a timestamp is used.",
    )
    update_most_recent_symlink: bool = Field(
        default=True,
        description="Create/refresh a 'most-recent' symlink inside `root` after each run.",
    )
    z_axis_points_down: bool = Field(
        default=True,
        description=(
            "If true, negate world +Z in all exports and Rerun so the vertical axis **increases downward** "
            "(Z toward the ground) instead of a typical ENU/graphics +Z up convention."
        ),
    )


class ArucoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dictionary: str = Field(default="4x4_50", description="OpenCV ArUco dictionary name.")
    marker_edge_length_m: float = Field(
        default=0.025,
        gt=0.0,
        description="Physical edge length of a single ArUco marker, in meters.",
    )
    origin_marker_id: int | None = Field(
        default=None,
        description=(
            "If set, the ArUco frame's origin is the centroid of this marker's "
            "four 3D corners, and its X axis points along that marker's top "
            "edge. Otherwise the origin is the centroid of every triangulated corner."
        ),
    )
    min_views_per_marker: int = Field(
        default=2,
        ge=2,
        description="Skip markers that are seen in fewer than this many views.",
    )
    align_to_aruco: bool = Field(
        default=True,
        description=(
            "Enable ArUco-based Sim3 alignment after the back-end. When "
            "disabled, the reconstruction keeps its native frame."
        ),
    )
    max_sim3_scale_deviation_when_poses_known: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "When GT poses + intrinsics are provided, we refuse to apply the "
            "Sim3 scale if it deviates from 1.0 by more than this fraction. "
            "0.05 means '5% tolerance'."
        ),
    )
    detection_draw_scale: float = Field(
        default=2.0,
        gt=0.0,
        description="Thickness/size multiplier for annotation drawings.",
    )
    # Joint marker+camera bundle adjustment. Enforces rigid-square ArUco corners
    # (known edge length) and refines per-view camera poses so every detected
    # 2D corner reprojects to the stable 3D corner. Refined poses + scale are
    # propagated to the back-end's point maps (per-view) and the fused cloud is
    # regenerated, keeping the cloud, cameras, and markers mutually consistent.
    bundle_adjustment: bool = Field(
        default=True,
        description=(
            "Replace independent DLT triangulation with joint bundle "
            "adjustment over (marker 6-DoF, camera SE(3), metric scale). "
            "Required for ArUco 3D stability across views."
        ),
    )
    ba_max_iters: int = Field(
        default=300,
        ge=1,
        description="Max Adam iterations for the marker+camera bundle adjustment.",
    )
    ba_lr: float = Field(
        default=5e-3,
        gt=0.0,
        description="Adam learning rate for the marker+camera bundle adjustment.",
    )
    ba_huber_delta_px: float = Field(
        default=2.0,
        gt=0.0,
        description="Huber threshold on per-corner reprojection error, in pixels.",
    )
    ba_cam_prior_sigma_m: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "Soft Gaussian prior on per-view camera-center displacement "
            "(meters). Keeps the BA anchored to the back-end's initial poses "
            "when marker coverage is sparse; set larger if back-end poses are "
            "expected to be far off."
        ),
    )
    ba_cam_prior_sigma_deg: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Soft Gaussian prior on per-view camera rotation delta "
            "(degrees). Same role as `ba_cam_prior_sigma_m` for rotations."
        ),
    )
    # Drop outlier views after alignment using per-image marker reprojection RMSE.
    reject_views_by_alignment_error: bool = Field(
        default=False,
        description=(
            "After ArUco alignment, remove any view whose per-view corner "
            "reprojection RMSE exceeds `max_view_alignment_reproj_rmse_px`. "
            "The fused cloud, surface, saved poses, and Rerun cameras are "
            "rebuilt on kept views only (original indices recorded in metadata)."
        ),
    )
    max_view_alignment_reproj_rmse_px: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "Per-view RMSE threshold (pixels) for `reject_views_by_alignment_error`. "
            "Only views with at least one marker observation get a finite RMSE."
        ),
    )
    min_kept_views: int = Field(
        default=2,
        ge=2,
        description=(
            "Minimum number of views to keep; if rejection would go below this, "
            "all views are kept and a warning is printed."
        ),
    )
    reject_views_with_no_markers: bool = Field(
        default=False,
        description=(
            "When true, also reject views with no ArUco detections (RMSE undefined). "
            "Default false: those views still contribute dense geometry."
        ),
    )

    @field_validator("dictionary")
    @classmethod
    def _validate_dictionary(cls, value: str) -> str:
        if value not in ARUCO_DICTIONARIES:
            raise ValueError(
                f"Unknown ArUco dictionary {value!r}; "
                f"valid options: {sorted(ARUCO_DICTIONARIES.keys())}"
            )
        return value


class SurfaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_step: float = Field(
        default=0.0,
        ge=0.0,
        description="Height-field grid step in output units (<=0 = auto).",
    )
    fill_iters: int = Field(default=2, ge=0)
    smooth_iters: int = Field(default=1, ge=0)
    min_neighbors: int = Field(default=3, ge=1, le=9)
    max_resolution: int = Field(default=2048, ge=64)
    conf_threshold_percentile: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description=(
            "Confidence percentile used to filter points before surface "
            "reconstruction. 50 drops the bottom 50%."
        ),
    )


class Mast3rConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    pipeline_variant: Literal["dense", "sfm"] = Field(
        default="sfm",
        description=(
            "'dense' uses the DUSt3R modular global alignment path. "
            "'sfm' runs MASt3R-SfM (make_pairs + sparse_global_alignment)."
        ),
    )
    scene_graph: str = Field(
        default="auto",
        description=(
            "MASt3R-SfM scene graph. 'auto' picks 'complete' for <40 images, "
            "else 'swin-5'. Explicit examples: 'complete', 'swin-5', "
            "'logwin-3', 'retrieval-20-1'."
        ),
    )
    retrieval_model: str | None = Field(
        default=None,
        description=(
            "Optional retrieval checkpoint for retrieval-based MASt3R-SfM "
            "pairing; required when scene_graph contains 'retrieval'."
        ),
    )
    sfm_subsample: int = Field(default=4, ge=1)
    sfm_lr1: float = Field(default=0.07, gt=0.0)
    sfm_niter1: int = Field(default=300, ge=0)
    sfm_lr2: float = Field(default=0.01, gt=0.0)
    sfm_niter2: int = Field(default=300, ge=0)
    sfm_opt_depth: bool = Field(default=True)
    sfm_shared_intrinsics: bool = Field(default=True)
    sfm_matching_conf_thr: float = Field(default=5.0, ge=0.0)
    sfm_min_conf_thr: float = Field(default=1.5, ge=0.0)
    sfm_clean_depth: bool = Field(default=True)
    image_size: int = Field(default=512, ge=64)
    neighbor_window: int = Field(default=2, ge=1)
    desc_conf_thr: float = Field(default=0.1, ge=0.0)
    dense_conf_thr: float = Field(default=12.0, ge=0.0)
    voxel_size: float = Field(default=0.0015, ge=0.0)
    max_points: int = Field(default=2_000_000, ge=1)
    pixel_tol: float = Field(default=1.5, ge=0.0)
    max_matches_per_pair: int = Field(default=50_000, ge=1)
    pose_refine_iters: int = Field(default=1, ge=0)
    pose_refine_lr: float = Field(default=0.0, ge=0.0)
    pose_refine_lr_min: float = Field(default=1e-5, ge=0.0)
    pose_prior_sigma_deg: float = Field(default=1.0, gt=0.0)
    pose_prior_sigma_m: float = Field(default=0.01, gt=0.0)
    pose_prior_weight: float = Field(default=0.06, ge=0.0)
    pose_refine_log_every: int = Field(default=25, ge=1)
    pose_refine_max_drot_deg: float = Field(default=8.0, ge=0.0)
    pose_refine_max_dt_m: float = Field(default=0.06, ge=0.0)
    dense_refine_iters: int = Field(default=1, ge=0)
    dense_refine_lr: float = Field(default=0.0, ge=0.0)
    confidence_percentile: float = Field(default=99.0, ge=0.0, le=100.0)


class RerunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    grpc_port: int = Field(default=9876, ge=1, le=65535)
    no_wait: bool = Field(default=False, description="Do not block on user input after logging.")
    log_cloud_rgb: bool = Field(default=True, description="Log fused RGB cloud under /points.")
    log_cloud_confidence: bool = Field(
        default=True,
        description="Log confidence-colored cloud under /points_confidence.",
    )
    log_cameras: bool = Field(default=True, description="Log camera transforms + pinhole models.")
    log_camera_aruco_images: bool = Field(
        default=True,
        description="Log per-camera ArUco-annotated images under /cameras/*/image/aruco.",
    )
    log_camera_confidence_images: bool = Field(
        default=True,
        description="Log per-camera confidence images under /cameras/*/image/confidence.",
    )
    log_aruco_3d: bool = Field(default=True, description="Log aligned 3D ArUco markers and plane.")
    log_surface_mesh: bool = Field(default=True, description="Log reconstructed mesh under /surface/mesh.")
    log_surface_cloud_open3d_web: bool = Field(
        default=False,
        description=(
            "Also open the reconstructed surface point cloud in Open3D's web visualizer "
            "(uses `open3d.visualization.draw`)."
        ),
    )
    open3d_web_visualizer_port: int = Field(
        default=8888,
        ge=1,
        le=65535,
        description=(
            "Port for the Open3D web visualizer when enabled. Applied by setting the "
            "`WEBRTC_PORT` environment variable before `open3d.visualization.draw`."
        ),
    )
    open3d_web_show_ui: bool = Field(
        default=True,
        description="Whether to show Open3D's UI controls in the web visualizer.",
    )
    pointcloud_resample_factor: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of cloud points to log to Rerun (1.0 logs all points).",
    )
    save_rrd: bool = Field(
        default=True,
        description="If true, stream the recording to a .rrd file (usually under the run’s rerun/ folder).",
    )
    save_rbl: bool = Field(
        default=True,
        description="If true, save the viewer layout blueprint as .rbl next to the .rrd (requires rerun-sdk with Blueprint.save).",
    )
    rerun_subdir: str = Field(
        default="rerun",
        description="Subfolder under the run output for .rrd and .rbl when saving.",
    )
    rrd_basename: str = Field(
        default="spectra.rrd",
        description="Rerun recording file name; should end in .rrd.",
    )
    rbl_basename: str = Field(
        default="spectra.rbl",
        description="Rerun blueprint file name; should end in .rbl.",
    )


class ReconstructionConfig(BaseModel):
    """Top-level configuration for `run_reconstruction`."""

    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    aruco: ArucoConfig = Field(default_factory=ArucoConfig)
    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    mast3r: Mast3rConfig = Field(default_factory=Mast3rConfig)
    rerun: RerunConfig = Field(default_factory=RerunConfig)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "ReconstructionConfig":
        """Resolve any relative paths to absolute, anchored at cwd."""
        self.input.rgb_dir = Path(self.input.rgb_dir).expanduser()
        if self.input.pose_dir is not None:
            self.input.pose_dir = Path(self.input.pose_dir).expanduser()
        if self.input.camera_params_dir is not None:
            self.input.camera_params_dir = Path(self.input.camera_params_dir).expanduser()
        self.output.root = Path(self.output.root).expanduser()
        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a pure-python dict with `Path` objects converted to strings."""
        return _jsonable(self.model_dump(mode="python"))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ReconstructionConfig":
        """Return a new config with dotted-path overrides applied.

        Example::

            cfg.with_overrides({
                "aruco.marker_edge_length_m": 0.03,
                "mast3r.voxel_size": 0.002,
            })
        """
        base = copy.deepcopy(self.model_dump(mode="python"))
        for dotted_key, value in overrides.items():
            _set_dotted(base, dotted_key, value)
        return ReconstructionConfig.model_validate(base)


def _jsonable(obj: Any) -> Any:
    """Recursively convert `Path` to `str` and pydantic submodels to dicts."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, BaseModel):
        return _jsonable(obj.model_dump(mode="python"))
    return obj


def _set_dotted(root: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Any = root
    for part in parts[:-1]:
        if not isinstance(cursor, MutableMapping) or part not in cursor:
            raise KeyError(f"Unknown config path {dotted_key!r} at segment {part!r}")
        cursor = cursor[part]
    if not isinstance(cursor, MutableMapping) or parts[-1] not in cursor:
        raise KeyError(f"Unknown config path {dotted_key!r}")
    cursor[parts[-1]] = value


def load_config(path: str | Path) -> ReconstructionConfig:
    """Load a YAML file and validate it into a `ReconstructionConfig`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return ReconstructionConfig.model_validate(data)


def save_config(cfg: ReconstructionConfig, path: str | Path) -> Path:
    """Serialize a config to YAML and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_yaml_dict(), f, sort_keys=False, default_flow_style=False)
    return path


def save_config_json(cfg: ReconstructionConfig, path: str | Path) -> Path:
    """Serialize a config to JSON (kept alongside the YAML in run folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_yaml_dict(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ArucoConfig",
    "InputConfig",
    "Mast3rConfig",
    "OutputConfig",
    "ReconstructionConfig",
    "RerunConfig",
    "SurfaceConfig",
    "load_config",
    "save_config",
    "save_config_json",
]
