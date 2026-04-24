"""Backward-compatible CLI shim for the VGGT backend.

The full reconstruction logic now lives in ``spectra.backends.vggt_backend``
and the orchestrator in ``spectra.pipeline``. This script keeps the legacy
command-line interface working by parsing the old flags, building a
`ReconstructionConfig`, and invoking `run_reconstruction`.

For the new unified entry point, see ``python -m spectra reconstruct --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectra import ReconstructionConfig, run_reconstruction
from spectra.config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    RerunConfig,
    SurfaceConfig,
    VggtConfig,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT Point Cloud Extraction (legacy CLI shim)")
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction_vggt"))

    p.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--conf_thres", type=float, default=50.0)
    p.add_argument("--cloud_source", type=str, choices=["point_map", "depth_map"], default="depth_map")
    p.add_argument("--camera_source", type=str, choices=["predicted", "gt"], default="predicted")
    p.add_argument("--alignment_mode", type=str, choices=["sim3", "se3"], default="sim3")
    p.add_argument("--mask_black_bg", action="store_true")
    p.add_argument("--mask_white_bg", action="store_true")

    p.add_argument("--surface_grid_step", type=float, default=0.0)
    p.add_argument("--surface_fill_iters", type=int, default=2)
    p.add_argument("--surface_smooth_iters", type=int, default=1)
    p.add_argument("--surface_min_neighbors", type=int, default=3)
    p.add_argument("--surface_max_resolution", type=int, default=2048)

    p.add_argument("--aruco_dict", type=str, default="4x4_50")
    p.add_argument("--aruco_edge_length_m", type=float, default=0.025)
    p.add_argument("--no_aruco", action="store_true", help="Skip ArUco alignment.")

    p.add_argument("--grpc_port", type=int, default=9876)
    p.add_argument("--no_wait", action="store_true")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> ReconstructionConfig:
    pose_dir = args.pose_dir if args.pose_dir.exists() else None
    camera_params_dir = args.camera_params_dir if args.camera_params_dir.exists() else None

    return ReconstructionConfig(
        input=InputConfig(
            rgb_dir=args.rgb_dir,
            pose_dir=pose_dir,
            camera_params_dir=camera_params_dir,
        ),
        output=OutputConfig(root=args.out_dir),
        backend="vggt",
        aruco=ArucoConfig(
            dictionary=args.aruco_dict,
            marker_edge_length_m=args.aruco_edge_length_m,
            align_to_aruco=not args.no_aruco,
        ),
        surface=SurfaceConfig(
            grid_step=args.surface_grid_step,
            fill_iters=args.surface_fill_iters,
            smooth_iters=args.surface_smooth_iters,
            min_neighbors=args.surface_min_neighbors,
            max_resolution=args.surface_max_resolution,
        ),
        vggt=VggtConfig(
            model_name=args.model_name,
            image_size=args.image_size,
            conf_thres=args.conf_thres,
            cloud_source=args.cloud_source,
            camera_source=args.camera_source,
            alignment_mode=args.alignment_mode,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
        ),
        mast3r=Mast3rConfig(),
        rerun=RerunConfig(grpc_port=args.grpc_port, no_wait=args.no_wait),
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)
    run_reconstruction(cfg)


if __name__ == "__main__":
    main()
