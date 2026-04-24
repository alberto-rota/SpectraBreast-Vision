"""Typed configuration schema for the ArUco-stabilized reconstruction pipeline.

Configuration is stored as YAML on disk and round-tripped through pydantic v2
models so that CLI overrides, TUI edits, and API callers all see a single
validated object.

Typical usage::

    from spectra.config import load_config, ReconstructionConfig

    cfg = load_config("configs/default.yaml")
    cfg = cfg.with_overrides({"aruco.marker_edge_length_m": 0.03})

CLI overrides use dotted paths, e.g. ``--set aruco.marker_edge_length_m=0.03``
or ``--aruco.marker_edge_length_m 0.03`` depending on the CLI layer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .aruco import ARUCO_DICTIONARIES


Backend = Literal["vggt", "mast3r"]


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


class VggtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="facebook/VGGT-1B")
    image_size: int = Field(default=518, ge=64)
    conf_thres: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Confidence percentile used to filter VGGT points.",
    )
    cloud_source: Literal["depth_map", "point_map"] = Field(default="depth_map")
    camera_source: Literal["predicted", "gt"] = Field(
        default="predicted",
        description=(
            "When GT poses are available, 'gt' applies a pred->GT Sim3 so the "
            "cloud lives in the GT frame before ArUco alignment."
        ),
    )
    alignment_mode: Literal["sim3", "se3"] = Field(default="sim3")
    mask_black_bg: bool = Field(default=False)
    mask_white_bg: bool = Field(default=False)


class Mast3rConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
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


class ReconstructionConfig(BaseModel):
    """Top-level configuration for `run_reconstruction`."""

    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    backend: Backend = Field(default="vggt")
    aruco: ArucoConfig = Field(default_factory=ArucoConfig)
    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    vggt: VggtConfig = Field(default_factory=VggtConfig)
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
                "backend": "mast3r",
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
    "Backend",
    "InputConfig",
    "Mast3rConfig",
    "OutputConfig",
    "ReconstructionConfig",
    "RerunConfig",
    "SurfaceConfig",
    "VggtConfig",
    "load_config",
    "save_config",
    "save_config_json",
]
