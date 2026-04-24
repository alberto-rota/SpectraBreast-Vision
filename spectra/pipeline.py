"""Top-level orchestrator: detect ArUcos, run a back-end, align, surface.

Usage::

    from spectra import run_reconstruction, load_config

    cfg = load_config("configs/default.yaml")
    result = run_reconstruction(cfg)
    print(result.run_dir)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from rich import print

from .align import (
    ArucoAlignment,
    PlaneFrame,
    align_with_aruco,
    apply_similarity_to_camera_poses,
    apply_similarity_to_points,
)
from .aruco import ArucoDetector, MarkerDetection, annotate_image
from .config import ReconstructionConfig, save_config, save_config_json
from .inputs import build_backend_inputs
from .io_utils import prepare_run_dir, save_mesh_as_ply, save_pointcloud_as_ply, update_most_recent_symlink
from .rerun_logging import (
    HAS_RERUN,
    init_rerun,
    log_aruco_alignment_in_output_frame,
    log_camera,
    log_cloud,
    log_surface_mesh,
    rerun_flush,
    send_blueprint,
)
from .surface import SurfaceResult, reconstruct_surface


@dataclass
class ReconstructionResult:
    run_dir: Path
    cloud_points: np.ndarray
    cloud_colors: np.ndarray
    cloud_confidence: np.ndarray
    surface: SurfaceResult
    T_world_cam: np.ndarray
    K_per_view_orig: np.ndarray
    aruco_alignment: ArucoAlignment | None
    frame_description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _log_step(msg: str) -> None:
    print(f"[bold cyan]{msg}[/bold cyan]")


def _colorize_scalar_field(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vis = np.log1p(np.clip(arr, 0.0, None))
    finite = np.isfinite(vis)
    if not np.any(finite):
        return np.zeros(arr.shape + (3,), dtype=np.uint8)
    vmax = float(np.percentile(vis[finite], percentile))
    if vmax <= 1e-12:
        vmax = float(vis[finite].max())
    if vmax <= 1e-12:
        return np.zeros(arr.shape + (3,), dtype=np.uint8)
    norm = np.clip(vis / vmax, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    if u8.ndim == 1:
        colored = cv2.applyColorMap(u8[:, None], cv2.COLORMAP_TURBO)[:, 0, :]
        colored = colored[:, ::-1]
        colored[~finite] = 0
        return colored
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[~finite] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _detect_aruco_on_all(
    image_paths: List[Path],
    run_dir: Path,
    dictionary: str,
    draw_scale: float,
) -> tuple[List[List[MarkerDetection]], List[np.ndarray]]:
    """Run ArUco detection on every input image, saving annotated copies to disk.

    Returns:
        detections_per_view: list of lists of detections (indexed by view).
        annotated_rgb_per_view: RGB copies of the annotated images (for Rerun).
    """
    detector = ArucoDetector(dictionary=dictionary)
    aruco_dir = run_dir / "aruco_detections"
    json_dir = aruco_dir / "json"
    annotated_dir = aruco_dir / "annotated"
    json_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    detections_per_view: List[List[MarkerDetection]] = []
    annotated_rgb_per_view: List[np.ndarray] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            detections_per_view.append([])
            annotated_rgb_per_view.append(np.zeros((1, 1, 3), dtype=np.uint8))
            continue

        detections = detector.detect(image_bgr)
        detections_per_view.append(detections)

        annotated_bgr = annotate_image(image_bgr, detections, draw_scale=draw_scale)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb_per_view.append(annotated_rgb)

        cv2.imwrite(str(annotated_dir / image_path.name), annotated_bgr)
        payload = {
            "image_name": image_path.name,
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }
        (json_dir / f"{image_path.stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return detections_per_view, annotated_rgb_per_view


def _write_aruco_markers_3d(
    alignment: ArucoAlignment,
    edge_length_m: float,
    path: Path,
) -> None:
    payload: Dict[str, Any] = {
        "edge_length_m": float(edge_length_m),
        "scale_input_to_meters": float(alignment.scale),
        "scale_mad": float(alignment.scale_mad),
        "used_marker_ids": list(map(int, alignment.used_marker_ids)),
        "markers": {},
    }
    for marker_id, marker in alignment.markers.items():
        payload["markers"][str(int(marker_id))] = {
            "corners_3d": marker.corners_3d.tolist(),
            "num_views": int(marker.num_views),
            "reproj_rmse_px": float(marker.reproj_rmse_px),
            "edge_lengths_3d_input_frame": marker.edge_lengths_3d.tolist(),
            "center_3d": marker.center_3d.tolist(),
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dispatch_backend(cfg: ReconstructionConfig, inputs):
    if cfg.backend == "vggt":
        from .backends.vggt_backend import run_vggt
        return run_vggt(cfg, inputs)
    if cfg.backend == "mast3r":
        from .backends.mast3r_backend import run_mast3r
        return run_mast3r(cfg, inputs)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def run_reconstruction(cfg: ReconstructionConfig) -> ReconstructionResult:
    """Execute the full ArUco-stabilized reconstruction pipeline."""
    t0 = time.time()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _log_step("Step 1/6 - Preparing run folder and loading inputs")
    run_dir, run_name = prepare_run_dir(cfg.output.root, cfg.output.run_name)
    print(f"Run directory: [green]{run_dir}[/green]")

    backend_inputs = build_backend_inputs(
        rgb_dir=cfg.input.rgb_dir,
        pose_dir=cfg.input.pose_dir,
        camera_params_dir=cfg.input.camera_params_dir,
        device=device,
    )
    num_views = len(backend_inputs.image_paths)
    print(
        f"Loaded [green]{num_views}[/green] images"
        + (" with GT poses" if backend_inputs.T_world_cam_gt is not None else " (no GT poses)")
        + (" and GT intrinsics" if backend_inputs.K_orig_gt is not None else " (no GT intrinsics)")
    )

    if cfg.rerun.enabled and HAS_RERUN:
        init_rerun("Spectra_Reconstruction", grpc_port=cfg.rerun.grpc_port)

    _log_step("Step 2/6 - Detecting ArUco markers")
    detections_per_view, annotated_rgb_per_view = _detect_aruco_on_all(
        image_paths=backend_inputs.image_paths,
        run_dir=run_dir,
        dictionary=cfg.aruco.dictionary,
        draw_scale=cfg.aruco.detection_draw_scale,
    )
    num_detections = sum(len(d) for d in detections_per_view)
    print(f"Detected [green]{num_detections}[/green] markers across {num_views} views.")

    _log_step(f"Step 3/6 - Running backend [green]{cfg.backend}[/green]")
    raw = _dispatch_backend(cfg, backend_inputs)
    print(
        f"Backend produced [green]{raw.fused_points.shape[0]:,}[/green] fused points; "
        f"frame = [green]{raw.frame_description}[/green]"
    )

    aligned_cloud = raw.fused_points
    aligned_colors = raw.fused_colors
    aligned_confidence = raw.fused_confidence
    aligned_point_maps = raw.point_map_world
    aligned_poses = raw.T_world_cam
    plane_frame_output: PlaneFrame | None = None
    aruco_alignment: ArucoAlignment | None = None
    aligned_markers_in_output_frame: Dict[int, np.ndarray] = {}

    _log_step("Step 4/6 - Triangulating ArUco markers and building Sim3")
    poses_known = backend_inputs.T_world_cam_gt is not None and backend_inputs.K_orig_gt is not None
    if cfg.aruco.align_to_aruco and num_detections > 0:
        aruco_alignment = align_with_aruco(
            detections_per_view=detections_per_view,
            K_per_view=raw.K_per_view_orig,
            T_world_cam_per_view=raw.T_world_cam,
            edge_length_m=cfg.aruco.marker_edge_length_m,
            origin_marker_id=cfg.aruco.origin_marker_id,
            min_views_per_marker=cfg.aruco.min_views_per_marker,
            enforce_metric_scale=not poses_known,
        )

        if poses_known and abs(aruco_alignment.scale - 1.0) > cfg.aruco.max_sim3_scale_deviation_when_poses_known:
            print(
                f"[yellow]ArUco scale {aruco_alignment.scale:.4f} deviates from 1.0 "
                f"by more than {cfg.aruco.max_sim3_scale_deviation_when_poses_known:.2f}; "
                "poses are known so we refuse to rescale.[/yellow]"
            )

        if aruco_alignment.plane_frame is not None:
            s = aruco_alignment.sim3_scale
            R = aruco_alignment.rotation
            t = aruco_alignment.translation

            aligned_cloud = apply_similarity_to_points(raw.fused_points, s, R, t)
            aligned_point_maps = apply_similarity_to_points(raw.point_map_world, s, R, t)
            aligned_poses = apply_similarity_to_camera_poses(raw.T_world_cam, s, R, t)
            plane_frame_output = PlaneFrame(origin=np.zeros(3, dtype=np.float32), basis=np.eye(3, dtype=np.float32))
            aligned_markers_in_output_frame = {
                mid: apply_similarity_to_points(m.corners_3d, s, R, t)
                for mid, m in aruco_alignment.markers.items()
            }

            print(
                f"[green]ArUco alignment applied[/green]: "
                f"markers={len(aruco_alignment.markers)}, "
                f"scale={aruco_alignment.scale:.6f} (applied={s:.6f}), "
                f"MAD={aruco_alignment.scale_mad:.6f}"
            )
        else:
            print("[yellow]ArUco plane fit failed; keeping backend output frame.[/yellow]")
    else:
        if cfg.aruco.align_to_aruco:
            print("[yellow]No ArUco markers detected; skipping alignment.[/yellow]")

    _log_step("Step 5/6 - Reconstructing single-surface height field")
    surface = reconstruct_surface(
        fused_points=aligned_cloud,
        fused_colors=aligned_colors,
        fused_confidence=aligned_confidence,
        point_map_world=aligned_point_maps,
        valid_masks=raw.valid_masks,
        plane_frame=plane_frame_output,
        device=device,
        grid_step=cfg.surface.grid_step,
        fill_iters=cfg.surface.fill_iters,
        smooth_iters=cfg.surface.smooth_iters,
        min_neighbors=cfg.surface.min_neighbors,
        max_resolution=cfg.surface.max_resolution,
    )
    print(
        f"Surface: [green]{surface.points.shape[0]:,}[/green] vertices, "
        f"[green]{surface.triangles.shape[0]:,}[/green] triangles, "
        f"grid {surface.grid_shape[0]}x{surface.grid_shape[1]} @ step {surface.grid_step:.6f}"
    )

    _log_step("Step 6/6 - Writing outputs and logging to Rerun")
    _write_outputs(
        run_dir=run_dir,
        cfg=cfg,
        raw=raw,
        aligned_cloud=aligned_cloud,
        aligned_colors=aligned_colors,
        aligned_confidence=aligned_confidence,
        aligned_poses=aligned_poses,
        surface=surface,
        aruco_alignment=aruco_alignment,
        detections_per_view=detections_per_view,
    )

    if cfg.rerun.enabled and HAS_RERUN:
        _log_to_rerun(
            raw=raw,
            aligned_cloud=aligned_cloud,
            aligned_colors=aligned_colors,
            aligned_confidence=aligned_confidence,
            aligned_poses=aligned_poses,
            surface=surface,
            aruco_alignment=aruco_alignment,
            aligned_markers_in_output_frame=aligned_markers_in_output_frame,
            annotated_rgb_per_view=annotated_rgb_per_view,
            confidence_percentile=cfg.mast3r.confidence_percentile,
            poses_known=poses_known,
        )
        if not cfg.rerun.no_wait:
            input("Data has been logged to Rerun. Open the viewer, then press Enter to exit.")

    if cfg.output.update_most_recent_symlink:
        update_most_recent_symlink(cfg.output.root, run_dir)

    print(f"[green]Done in {time.time() - t0:.1f} s. Outputs: {run_dir}[/green]")

    return ReconstructionResult(
        run_dir=run_dir,
        cloud_points=aligned_cloud,
        cloud_colors=aligned_colors,
        cloud_confidence=aligned_confidence,
        surface=surface,
        T_world_cam=aligned_poses,
        K_per_view_orig=raw.K_per_view_orig,
        aruco_alignment=aruco_alignment,
        frame_description="aruco" if (aruco_alignment is not None and aruco_alignment.plane_frame is not None) else raw.frame_description,
        metadata={
            "backend": raw.backend_name,
            "run_name": run_name,
            "duration_s": time.time() - t0,
        },
    )


def _write_outputs(
    run_dir: Path,
    cfg: ReconstructionConfig,
    raw,
    aligned_cloud: np.ndarray,
    aligned_colors: np.ndarray,
    aligned_confidence: np.ndarray,
    aligned_poses: np.ndarray,
    surface: SurfaceResult,
    aruco_alignment: ArucoAlignment | None,
    detections_per_view: List[List[MarkerDetection]],
) -> None:
    save_pointcloud_as_ply(
        run_dir / "cloud.ply",
        aligned_cloud,
        aligned_colors,
        confidence=aligned_confidence,
    )
    np.savez(
        run_dir / "cloud.npz",
        points=aligned_cloud.astype(np.float32),
        colors=aligned_colors.astype(np.uint8),
        confidence=aligned_confidence.astype(np.float32),
    )

    save_pointcloud_as_ply(
        run_dir / "surface.ply",
        surface.points,
        surface.colors,
        confidence=surface.confidence,
        normals=surface.normals,
        support=surface.support,
    )
    save_mesh_as_ply(
        run_dir / "surface_mesh.ply",
        vertices=surface.points,
        triangles=surface.triangles,
        vertex_colors=surface.colors,
        vertex_normals=surface.normals,
    )
    np.savez(run_dir / "surface.npz", **surface.to_npz_dict())

    np.save(run_dir / "camera_poses_output_frame.npy", aligned_poses.astype(np.float32))
    np.save(run_dir / "intrinsics_original_frame.npy", raw.K_per_view_orig.astype(np.float32))
    np.save(run_dir / "intrinsics_network_frame.npy", raw.K_per_view_network.astype(np.float32))

    if aruco_alignment is not None and aruco_alignment.plane_frame is not None:
        np.save(run_dir / "sim3_to_output_frame.npy", aruco_alignment.sim3.astype(np.float32))
        _write_aruco_markers_3d(
            aruco_alignment,
            edge_length_m=cfg.aruco.marker_edge_length_m,
            path=run_dir / "aruco_markers_3d.json",
        )

    save_config(cfg, run_dir / "run.yaml")
    save_config_json(cfg, run_dir / "run.json")

    metadata = {
        "backend": raw.backend_name,
        "frame_description": raw.frame_description,
        "num_images": int(raw.T_world_cam.shape[0]),
        "num_cloud_points": int(aligned_cloud.shape[0]),
        "num_surface_points": int(surface.points.shape[0]),
        "num_surface_triangles": int(surface.triangles.shape[0]),
        "num_aruco_detections": int(sum(len(d) for d in detections_per_view)),
        "aruco_alignment": None,
        "alignment_info": raw.alignment_info,
    }
    if aruco_alignment is not None:
        metadata["aruco_alignment"] = {
            "num_markers": len(aruco_alignment.markers),
            "used_marker_ids": list(map(int, aruco_alignment.used_marker_ids)),
            "scale_input_to_meters": float(aruco_alignment.scale),
            "scale_mad": float(aruco_alignment.scale_mad),
            "sim3_scale_applied": float(aruco_alignment.sim3_scale),
            "warnings": aruco_alignment.warnings,
        }
    (run_dir / "reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _log_to_rerun(
    raw,
    aligned_cloud: np.ndarray,
    aligned_colors: np.ndarray,
    aligned_confidence: np.ndarray,
    aligned_poses: np.ndarray,
    surface: SurfaceResult,
    aruco_alignment: ArucoAlignment | None,
    aligned_markers_in_output_frame: Dict[int, np.ndarray],
    annotated_rgb_per_view: List[np.ndarray],
    confidence_percentile: float,
    poses_known: bool,
) -> None:
    log_cloud(
        aligned_cloud,
        aligned_colors,
        conf_colors=_colorize_scalar_field(aligned_confidence, percentile=confidence_percentile),
    )

    for idx in range(aligned_poses.shape[0]):
        K_orig = raw.K_per_view_orig[idx]
        orig_wh = (int(raw.original_image_sizes[idx, 0]), int(raw.original_image_sizes[idx, 1]))

        image_rgb = None
        image_aruco = None
        if idx < len(annotated_rgb_per_view):
            image_aruco = annotated_rgb_per_view[idx]
            image_rgb = image_aruco if image_aruco is not None else None

        conf_rgb = _colorize_scalar_field(raw.confidence_maps_network[idx], percentile=confidence_percentile)

        log_camera(
            index=idx,
            T_world_cam=aligned_poses[idx],
            K=K_orig,
            resolution_wh=orig_wh,
            image_rgb=None,
            image_aruco_rgb=image_aruco,
            confidence_rgb=conf_rgb,
        )

    if aruco_alignment is not None and aligned_markers_in_output_frame:
        log_aruco_alignment_in_output_frame(aligned_markers_in_output_frame)

    log_surface_mesh(
        vertices=surface.points,
        triangles=surface.triangles,
        vertex_colors=surface.colors,
        vertex_normals=surface.normals,
    )

    send_blueprint()
    rerun_flush()


__all__ = ["ReconstructionResult", "run_reconstruction"]
