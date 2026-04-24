"""Backward-compatible CLI shim for the MASt3R backend.

The full reconstruction logic now lives in ``spectra.backends.mast3r_backend``
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
    p = argparse.ArgumentParser(
        description="Metric dense MASt3R fusion (legacy CLI shim)"
    )
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction"))

    p.add_argument("--model_name", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--neighbor_window", type=int, default=2)

    p.add_argument("--desc_conf_thr", type=float, default=0.1)
    p.add_argument("--pixel_tol", type=float, default=1.5)
    p.add_argument("--max_matches_per_pair", type=int, default=50_000)
    p.add_argument("--pose_refine_iters", type=int, default=1)
    p.add_argument("--pose_refine_lr", type=float, default=0.0)
    p.add_argument("--pose_refine_lr_min", type=float, default=1e-5)
    p.add_argument("--pose_prior_sigma_deg", type=float, default=1.0)
    p.add_argument("--pose_prior_sigma_m", type=float, default=0.01)
    p.add_argument("--pose_prior_weight", type=float, default=0.06)
    p.add_argument("--pose_refine_log_every", type=int, default=25)
    p.add_argument("--pose_refine_max_drot_deg", type=float, default=8.0)
    p.add_argument("--pose_refine_max_dt_m", type=float, default=0.06)

    p.add_argument("--dense_refine_iters", type=int, default=1)
    p.add_argument("--dense_refine_lr", type=float, default=0.0)
    p.add_argument("--dense_conf_thr", type=float, default=12.0)
    p.add_argument("--confidence_percentile", type=float, default=99.0)
    p.add_argument("--voxel_size", type=float, default=0.0015)
    p.add_argument("--max_points", type=int, default=2_000_000)

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
        backend="mast3r",
        aruco=ArucoConfig(
            dictionary=args.aruco_dict,
            marker_edge_length_m=args.aruco_edge_length_m,
            align_to_aruco=not args.no_aruco,
        ),
        surface=SurfaceConfig(),
        vggt=VggtConfig(),
        mast3r=Mast3rConfig(
            model_name=args.model_name,
            image_size=args.image_size,
            neighbor_window=args.neighbor_window,
            desc_conf_thr=args.desc_conf_thr,
            pixel_tol=args.pixel_tol,
            max_matches_per_pair=args.max_matches_per_pair,
            pose_refine_iters=args.pose_refine_iters,
            pose_refine_lr=args.pose_refine_lr,
            pose_refine_lr_min=args.pose_refine_lr_min,
            pose_prior_sigma_deg=args.pose_prior_sigma_deg,
            pose_prior_sigma_m=args.pose_prior_sigma_m,
            pose_prior_weight=args.pose_prior_weight,
            pose_refine_log_every=args.pose_refine_log_every,
            pose_refine_max_drot_deg=args.pose_refine_max_drot_deg,
            pose_refine_max_dt_m=args.pose_refine_max_dt_m,
            dense_refine_iters=args.dense_refine_iters,
            dense_refine_lr=args.dense_refine_lr,
            dense_conf_thr=args.dense_conf_thr,
            confidence_percentile=args.confidence_percentile,
            voxel_size=args.voxel_size,
            max_points=args.max_points,
        ),
        rerun=RerunConfig(grpc_port=args.grpc_port, no_wait=args.no_wait),
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)
    run_reconstruction(cfg)


if __name__ == "__main__":
    main()
