"""Top-level orchestrator: detect ArUcos, run a back-end, align, surface.

Usage::

    from spectra import run_reconstruction, load_config

    result = run_reconstruction(load_config("configs/default.yaml"))
    print(result.run_dir)
"""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field, replace
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
    marker_corner_reprojection_stats,
    markers_best_fit_plane_rms_m,
    per_view_marker_corner_rmse_px,
)
from .backends.types import RawReconstruction
from .aruco import ArucoDetector, MarkerDetection, annotate_image
from .config import ReconstructionConfig, RerunConfig, save_config, save_config_json
from .coordinates import apply_output_z_reflection, reflect_z_sim3_matrix
from .inputs import build_backend_inputs, canonicalize_images_with_exif
from .io_utils import prepare_run_dir, save_mesh_as_ply, save_pointcloud_as_ply, update_most_recent_symlink
from .marker_ba import BundleAdjustmentResult, apply_delta_and_scale_to_points
from .rerun_logging import (
    HAS_RERUN,
    init_rerun,
    log_aruco_alignment_in_output_frame,
    log_camera,
    log_cloud,
    log_surface_mesh,
    resample_cloud_for_logging,
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


def _log_surface_cloud_open3d_web(
    surface: SurfaceResult,
    web_visualizer_port: int,
    show_ui: bool,
) -> None:
    """Display the reconstructed surface cloud in Open3D's web viewer."""
    if surface.points.shape[0] == 0:
        print("[yellow]Skipping Open3D web viewer: surface cloud is empty.[/yellow]")
        return
    try:
        o3d = importlib.import_module("open3d")
    except Exception as exc:
        print(f"[yellow]Skipping Open3D web viewer: failed to import open3d ({exc}).[/yellow]")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface.points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        (surface.colors.astype(np.float64) / 255.0).clip(0.0, 1.0)
    )

    # `draw()` has no `web_visualizer_port` kwarg; the web server uses WEBRTC_*
    # (see https://www.open3d.org/docs/release/tutorial/visualization/web_visualizer.html).
    port = int(web_visualizer_port)
    os.environ["WEBRTC_PORT"] = str(port)

    o3d.visualization.draw(
        [pcd],
        show_ui=bool(show_ui),
    )
    print(
        "[green]Open3D web viewer launched[/green] for surface cloud at "
        f"http://127.0.0.1:{port}"
    )


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

    ``image_paths`` is expected to already be in EXIF-canonical orientation
    (see :func:`spectra.inputs.canonicalize_images_with_exif`), so this
    function does no EXIF handling itself — any consumer that opens those same
    files with OpenCV / PIL / torchvision will see identical pixels.

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
    corners_3d_output_frame: Dict[int, np.ndarray] | None = None,
) -> None:
    """Write marker geometry; ``corners_3d`` match ``cloud.ply`` when output frame is passed."""
    payload: Dict[str, Any] = {
        "edge_length_m": float(edge_length_m),
        "scale_input_to_meters": float(alignment.scale),
        "scale_mad": float(alignment.scale_mad),
        "used_marker_ids": list(map(int, alignment.used_marker_ids)),
        "frame": "output" if corners_3d_output_frame is not None else "pre_sim3_ba_metric",
        "markers": {},
    }
    for marker_id, marker in alignment.markers.items():
        mid = int(marker_id)
        corners_out = (
            np.asarray(corners_3d_output_frame[mid], dtype=np.float32).reshape(4, 3)
            if corners_3d_output_frame is not None and mid in corners_3d_output_frame
            else marker.corners_3d
        )
        edges_out = np.linalg.norm(
            np.roll(corners_out, -1, axis=0) - corners_out, axis=1
        ).astype(np.float32)
        center_out = corners_out.mean(axis=0).astype(np.float32)
        payload["markers"][str(mid)] = {
            "corners_3d": corners_out.tolist(),
            "num_views": int(marker.num_views),
            "reproj_rmse_px": float(marker.reproj_rmse_px),
            # Legacy field name; values match ``corners_3d`` (rigid transform preserves edge lengths).
            "edge_lengths_3d_input_frame": edges_out.tolist(),
            "center_3d": center_out.tolist(),
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_mast3r_backend(cfg: ReconstructionConfig, inputs) -> RawReconstruction:
    from .backends.mast3r_backend import run_mast3r

    return run_mast3r(cfg, inputs)


def _refuse_with_ba_delta(
    point_map_world: np.ndarray,          # [V, Hn, Wn, 3] backend-scale world
    valid_masks: np.ndarray,              # [V, Hn, Wn] bool
    images_network_uint8: np.ndarray,     # [V, Hn, Wn, 3] uint8
    confidence_maps_network: np.ndarray,  # [V, Hn, Wn] float32
    delta_T_per_view: np.ndarray,         # [V, 4, 4] float32 (metric)
    scale_m_per_backend: float,
    voxel_size: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-view metric delta + scale to per-pixel point maps and re-fuse.

    Returns:
        refined_point_map_world: [V, Hn, Wn, 3] float32 (metric, refined)
        fused_points:            [N, 3] float32
        fused_colors:            [N, 3] uint8
        fused_confidence:        [N]   float32
    """
    V, Hn, Wn, _ = point_map_world.shape
    view_idx_grid = np.broadcast_to(                          # [V, Hn, Wn] int32
        np.arange(V, dtype=np.int32)[:, None, None],
        (V, Hn, Wn),
    )

    refined_pm = apply_delta_and_scale_to_points(
        points_backend_world=point_map_world,
        view_idx_of_points=view_idx_grid,
        delta_T_per_view=delta_T_per_view,
        scale_m_per_backend=scale_m_per_backend,
    )                                                          # [V, Hn, Wn, 3] float32

    if not np.any(valid_masks):
        return (
            refined_pm,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    fused_points = refined_pm[valid_masks].astype(np.float32)
    fused_colors = images_network_uint8[valid_masks].astype(np.uint8)
    fused_confidence = confidence_maps_network[valid_masks].astype(np.float32)

    if voxel_size > 0.0 and fused_points.shape[0] > 0:
        # Simple confidence-weighted voxel downsample (kept local to avoid a
        # backends -> pipeline cycle; mirrors the MASt3R back-end's helper).
        keys = np.floor(fused_points / float(voxel_size)).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)  # numpy>=2.0 may return shape [N,1] for axis=0
        num_vox = int(inv.max()) + 1
        weights = np.clip(fused_confidence.astype(np.float64), 1e-6, None)
        weight_sum = np.bincount(inv, weights=weights, minlength=num_vox)
        counts = np.bincount(inv, minlength=num_vox)

        out_pts = np.zeros((num_vox, 3), dtype=np.float64)
        out_cols = np.zeros((num_vox, 3), dtype=np.float64)
        for d in range(3):
            out_pts[:, d] = np.bincount(
                inv, weights=weights * fused_points[:, d].astype(np.float64), minlength=num_vox
            ) / np.maximum(weight_sum, 1e-12)
            out_cols[:, d] = np.bincount(
                inv, weights=weights * fused_colors[:, d].astype(np.float64), minlength=num_vox
            ) / np.maximum(weight_sum, 1e-12)
        out_conf = np.bincount(
            inv, weights=fused_confidence.astype(np.float64), minlength=num_vox
        ) / np.maximum(counts, 1)

        fused_points = out_pts.astype(np.float32)
        fused_colors = np.clip(out_cols, 0.0, 255.0).astype(np.uint8)
        fused_confidence = out_conf.astype(np.float32)

    return refined_pm, fused_points, fused_colors, fused_confidence


def _fuse_point_maps_to_cloud(
    point_maps: np.ndarray,
    valid_masks: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse per-view maps into a single cloud with one vectorized gather."""
    mask = np.asarray(valid_masks, dtype=bool)
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        point_maps[mask].astype(np.float32),  # [N, 3]
        colors[mask].astype(np.uint8),        # [N, 3]
        conf[mask].astype(np.float32),        # [N]
    )


def _slice_raw_reconstruction(raw: RawReconstruction, keep_idx: np.ndarray) -> RawReconstruction:
    """Restrict every per-view field to ``keep_idx`` and refresh ``fused_*`` from maps."""
    ki = np.asarray(keep_idx, dtype=np.int64)
    pm = raw.point_map_world[ki]
    vm = raw.valid_masks[ki]
    imgs = raw.images_network_uint8[ki]
    cmap = raw.confidence_maps_network[ki]
    fused_pts, fused_col, fused_cf = _fuse_point_maps_to_cloud(pm, vm, imgs, cmap)
    extra = dict(raw.extra)
    geoms = extra.get("geometries")
    if isinstance(geoms, list):
        extra["geometries"] = [geoms[int(i)] for i in ki]
    for key in ("T_world_cam_predicted", "intrinsics_predicted"):
        arr = extra.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == raw.T_world_cam.shape[0]:
            extra[key] = arr[ki]
    return replace(
        raw,
        fused_points=fused_pts,
        fused_colors=fused_col,
        fused_confidence=fused_cf,
        point_map_world=pm,
        valid_masks=vm,
        T_world_cam=raw.T_world_cam[ki],
        K_per_view_orig=raw.K_per_view_orig[ki],
        K_per_view_network=raw.K_per_view_network[ki],
        network_image_sizes=raw.network_image_sizes[ki],
        original_image_sizes=raw.original_image_sizes[ki],
        images_network_uint8=imgs,
        confidence_maps_network=cmap,
        extra=extra,
    )


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

    # Bake EXIF Orientation into pixel buffers once, then reuse those paths
    # everywhere downstream. MASt3R and the ArUco
    # detector all open files via cv2 / PIL / torchvision, which ignore EXIF
    # by default — so mixing upright and rotated/mirrored raw buffers (as
    # iPhone JPEGs do within a single batch) would otherwise feed backends
    # images they were never trained on, producing wildly inconsistent poses.
    canonical_rgb_dir = run_dir / "inputs_exif_canonical"
    canonical_paths = canonicalize_images_with_exif(
        image_paths=backend_inputs.image_paths,
        cache_dir=canonical_rgb_dir,
    )
    backend_inputs.image_paths = canonical_paths

    if cfg.rerun.enabled and HAS_RERUN:
        rdir = (run_dir / cfg.rerun.rerun_subdir).resolve()
        if cfg.rerun.save_rrd or cfg.rerun.save_rbl:
            rdir.mkdir(parents=True, exist_ok=True)
        rrd_path: Path | None = (rdir / cfg.rerun.rrd_basename).resolve() if cfg.rerun.save_rrd else None
        init_rerun("Spectra_Reconstruction", int(cfg.rerun.grpc_port), rrd_path=rrd_path)

    _log_step("Step 2/6 - Detecting ArUco markers")
    detections_per_view, annotated_rgb_per_view = _detect_aruco_on_all(
        image_paths=backend_inputs.image_paths,
        run_dir=run_dir,
        dictionary=cfg.aruco.dictionary,
        draw_scale=cfg.aruco.detection_draw_scale,
    )
    num_detections = sum(len(d) for d in detections_per_view)
    print(f"Detected [green]{num_detections}[/green] markers across {num_views} views.")

    _log_step("Step 3/6 - Running MASt3R-SfM")
    raw = _run_mast3r_backend(cfg, backend_inputs)
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
    aruco_geometric_qc: Dict[str, Any] | None = None
    view_rejection_meta: Dict[str, Any] | None = None

    poses_known = backend_inputs.T_world_cam_gt is not None and backend_inputs.K_orig_gt is not None

    use_ba = bool(cfg.aruco.bundle_adjustment) and (num_detections > 0)
    if cfg.aruco.align_to_aruco and num_detections > 0:
        if use_ba:
            _log_step(
                "Step 4/6 - Jointly bundle-adjusting ArUco markers + cameras (rigid squares)"
            )
        else:
            _log_step("Step 4/6 - Triangulating ArUco markers and building Sim3")
        aruco_alignment = align_with_aruco(
            detections_per_view=detections_per_view,
            K_per_view=raw.K_per_view_orig,
            T_world_cam_per_view=raw.T_world_cam,
            edge_length_m=cfg.aruco.marker_edge_length_m,
            origin_marker_id=cfg.aruco.origin_marker_id,
            min_views_per_marker=cfg.aruco.min_views_per_marker,
            enforce_metric_scale=not poses_known,
            use_bundle_adjustment=use_ba,
            ba_options={
                "max_iters": cfg.aruco.ba_max_iters,
                "lr": cfg.aruco.ba_lr,
                "huber_delta_px": cfg.aruco.ba_huber_delta_px,
                "cam_prior_sigma_m": cfg.aruco.ba_cam_prior_sigma_m,
                "cam_prior_sigma_deg": cfg.aruco.ba_cam_prior_sigma_deg,
            } if use_ba else None,
        )

        # For a BA run, the returned `scale` is always 1.0 (frame is in meters);
        # this deviation check only makes sense on the DLT path.
        if (
            not use_ba
            and poses_known
            and abs(aruco_alignment.scale - 1.0) > cfg.aruco.max_sim3_scale_deviation_when_poses_known
        ):
            print(
                f"[yellow]ArUco scale {aruco_alignment.scale:.4f} deviates from 1.0 "
                f"by more than {cfg.aruco.max_sim3_scale_deviation_when_poses_known:.2f}; "
                "poses are known so we refuse to rescale.[/yellow]"
            )

        if aruco_alignment.plane_frame is not None:
            ba = aruco_alignment.bundle_adjustment  # type: BundleAdjustmentResult | None

            # (1) First bring the backend cloud + cameras into the BA's metric frame
            # (per-view SE(3) delta + shared scale), then (2) apply the Sim3 that
            # places the ArUco plane at Z=0. Step (1) is the key invariant that
            # makes every detected ArUco corner reproject to the stable 3D corner.
            if ba is not None and ba.marker_corners_m:
                voxel_size = float(cfg.mast3r.voxel_size)
                (
                    refined_point_maps,
                    refused_points,
                    refused_colors,
                    refused_conf,
                ) = _refuse_with_ba_delta(
                    point_map_world=raw.point_map_world,
                    valid_masks=raw.valid_masks,
                    images_network_uint8=raw.images_network_uint8,
                    confidence_maps_network=raw.confidence_maps_network,
                    delta_T_per_view=ba.delta_T_per_view,
                    scale_m_per_backend=ba.scale_m_per_backend,
                    voxel_size=voxel_size,
                )
                backend_cloud_points = refused_points
                backend_cloud_colors = refused_colors
                backend_cloud_conf = refused_conf
                backend_point_maps = refined_point_maps
                backend_poses = ba.T_world_cam_refined  # already = Δ_v @ (scaled backend)
                print(
                    f"[green]Bundle adjustment[/green]: {len(ba.marker_corners_m)} markers, "
                    f"{ba.num_observations} obs, "
                    f"reproj RMSE {ba.initial_reproj_rmse_px:.3f} -> {ba.final_reproj_rmse_px:.3f} px"
                    f" ({ba.num_iters} iters{', converged' if ba.converged else ''})"
                )
                # Initial DLT RMSE is only meaningful before BA; ``final`` is what
                # matters for 3D coherence. Keep the scary warning for structural
                # breaks (hundreds of px initial), not for moderate pre-BA error
                # that BA then fixes.
                if ba.initial_reproj_rmse_px > 200.0:
                    print(
                        "[yellow]Warning: very large initial DLT reprojection error — "
                        "the ArUco detections and backend cameras likely live in "
                        "inconsistent coordinate systems (check intrinsics / image "
                        "orientation / padding).[/yellow]"
                    )
                elif ba.final_reproj_rmse_px > 12.0:
                    print(
                        "[yellow]Notice: BA reprojection RMSE is still high (>12 px); "
                        "markers may look soft vs the dense cloud. Try lowering "
                        "`aruco.ba_huber_delta_px`, tightening intrinsics, or "
                        "undistorting detections if the lens is strong.[/yellow]"
                    )
            else:
                backend_cloud_points = raw.fused_points
                backend_cloud_colors = raw.fused_colors
                backend_cloud_conf = raw.fused_confidence
                backend_point_maps = raw.point_map_world
                backend_poses = raw.T_world_cam

            s = aruco_alignment.sim3_scale
            R = aruco_alignment.rotation
            t = aruco_alignment.translation

            aligned_cloud = apply_similarity_to_points(backend_cloud_points, s, R, t)
            aligned_colors = backend_cloud_colors
            aligned_confidence = backend_cloud_conf
            aligned_point_maps = apply_similarity_to_points(backend_point_maps, s, R, t)
            aligned_poses = apply_similarity_to_camera_poses(backend_poses, s, R, t)
            plane_frame_output = PlaneFrame(origin=np.zeros(3, dtype=np.float32), basis=np.eye(3, dtype=np.float32))
            aligned_markers_in_output_frame = {
                mid: apply_similarity_to_points(m.corners_3d, s, R, t)
                for mid, m in aruco_alignment.markers.items()
            }

            plane_rms_m = markers_best_fit_plane_rms_m(aligned_markers_in_output_frame)
            reproj_qc = marker_corner_reprojection_stats(
                detections_per_view=detections_per_view,
                marker_corners_world=aligned_markers_in_output_frame,
                K_per_view=raw.K_per_view_orig,
                T_world_cam=aligned_poses,
            )
            aruco_geometric_qc = {
                "markers_plane_rms_m": float(plane_rms_m),
                "markers_plane_rms_mm": float(plane_rms_m * 1000.0),
                **reproj_qc,
            }
            print(
                f"[dim]ArUco geometric QC:[/dim] best-fit plane RMS [cyan]{plane_rms_m * 1000:.2f} mm[/cyan]; "
                f"corners reprojected with output cameras → RMSE [cyan]{reproj_qc['rmse_px']:.3f} px[/cyan] "
                f"(max {reproj_qc['max_px']:.3f}, p95 {reproj_qc['p95_px']:.3f}, n={reproj_qc['num_observations']})"
            )
            if plane_rms_m > 0.015:
                print(
                    "[yellow]Markers deviate >15 mm from a single plane — if they are on one sheet, check BA / "
                    "intrinsics; otherwise multi-planar layouts are expected to show this.[/yellow]"
                )

            print(
                f"[green]ArUco alignment applied[/green]: "
                f"markers={len(aruco_alignment.markers)}, "
                f"scale={aruco_alignment.scale:.6f} (applied={s:.6f}), "
                f"MAD={aruco_alignment.scale_mad:.6f}"
            )
        else:
            print("[yellow]ArUco plane fit failed; keeping backend output frame.[/yellow]")
    else:
        _log_step("Step 4/6 - ArUco alignment disabled or no markers detected")
        if cfg.aruco.align_to_aruco:
            print("[yellow]No ArUco markers detected; skipping alignment.[/yellow]")

    V_all = int(raw.T_world_cam.shape[0])
    if (
        cfg.aruco.reject_views_by_alignment_error
        and aruco_alignment is not None
        and aruco_alignment.plane_frame is not None
        and aligned_markers_in_output_frame
        and aligned_poses.shape[0] == V_all
    ):
        err_v = per_view_marker_corner_rmse_px(
            detections_per_view=detections_per_view,
            marker_corners_world=aligned_markers_in_output_frame,
            K_per_view=raw.K_per_view_orig,
            T_world_cam=aligned_poses,
        )
        reject = np.zeros(V_all, dtype=bool)
        finite = np.isfinite(err_v)
        reject[finite & (err_v > float(cfg.aruco.max_view_alignment_reproj_rmse_px))] = True
        if cfg.aruco.reject_views_with_no_markers:
            reject[~finite] = True
        keep_idx = np.nonzero(~reject)[0]
        min_k = int(cfg.aruco.min_kept_views)
        if keep_idx.size < min_k:
            print(
                f"[yellow]View rejection skipped: would keep {keep_idx.size} views "
                f"(minimum {min_k}).[/yellow]"
            )
        elif bool(reject.any()):
            bad_idx = np.nonzero(reject)[0]
            thr = float(cfg.aruco.max_view_alignment_reproj_rmse_px)
            print(
                f"[yellow]Rejecting {bad_idx.size} view(s) with per-view marker RMSE > {thr} px "
                f"(0-based indices {bad_idx.tolist()}).[/yellow]"
            )
            ki = keep_idx
            aligned_point_maps = aligned_point_maps[ki]
            aligned_poses = aligned_poses[ki]
            raw = _slice_raw_reconstruction(raw, ki)
            aligned_cloud, aligned_colors, aligned_confidence = _fuse_point_maps_to_cloud(
                aligned_point_maps,
                raw.valid_masks,
                raw.images_network_uint8,
                raw.confidence_maps_network,
            )
            detections_per_view = [detections_per_view[int(i)] for i in ki]
            annotated_rgb_per_view = [annotated_rgb_per_view[int(i)] for i in ki]
            per_view_payload = {
                str(int(v)): float(err_v[v])
                for v in range(V_all)
                if bool(np.isfinite(err_v[v]))
            }
            view_rejection_meta = {
                "enabled": True,
                "max_view_alignment_reproj_rmse_px": thr,
                "rejected_original_indices": bad_idx.astype(int).tolist(),
                "kept_original_indices": ki.astype(int).tolist(),
                "kept_image_paths": [str(backend_inputs.image_paths[int(i)]) for i in ki],
                "per_view_reproj_rmse_px": per_view_payload,
            }
            if aruco_geometric_qc is not None:
                reproj_kept = marker_corner_reprojection_stats(
                    detections_per_view=detections_per_view,
                    marker_corners_world=aligned_markers_in_output_frame,
                    K_per_view=raw.K_per_view_orig,
                    T_world_cam=aligned_poses,
                )
                aruco_geometric_qc["after_view_rejection"] = {
                    "num_views": int(ki.size),
                    **{
                        k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                        for k, v in reproj_kept.items()
                    },
                }
                print(
                    f"[dim]After view rejection:[/dim] {ki.size} views; "
                    f"cloud [cyan]{aligned_cloud.shape[0]:,}[/cyan] points; "
                    f"marker reproj RMSE [cyan]{reproj_kept['rmse_px']:.3f} px[/cyan]"
                )
            else:
                print(
                    f"[dim]After view rejection:[/dim] {ki.size} views; "
                    f"cloud [cyan]{aligned_cloud.shape[0]:,}[/cyan] points"
                )
        else:
            view_rejection_meta = {
                "enabled": True,
                "max_view_alignment_reproj_rmse_px": float(cfg.aruco.max_view_alignment_reproj_rmse_px),
                "rejected_original_indices": [],
                "kept_original_indices": list(range(V_all)),
                "note": "no views exceeded threshold",
            }

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

    if cfg.output.z_axis_points_down:
        apply_output_z_reflection(
            aligned_cloud=aligned_cloud,
            aligned_poses=aligned_poses,
            aligned_markers_in_output_frame=aligned_markers_in_output_frame,
            surface=surface,
        )
        print("[dim]Output frame: world +Z points down (Z negated).[/dim]")

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
        aligned_markers_output_frame=aligned_markers_in_output_frame,
        aruco_geometric_qc=aruco_geometric_qc,
        view_rejection_meta=view_rejection_meta,
    )

    if cfg.rerun.enabled and HAS_RERUN:
        _log_to_rerun(
            run_dir=run_dir,
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
            rerun_cfg=cfg.rerun,
        )
        if not cfg.rerun.no_wait:
            input("Data has been logged to Rerun. Open the viewer, then press Enter to exit.")
    print("Checking if we should log the surface cloud to Open3D")
    if cfg.rerun.log_surface_cloud_open3d_web:
        print("Logging the surface cloud to Open3D")
        _log_surface_cloud_open3d_web(
            surface=surface,
            web_visualizer_port=cfg.rerun.open3d_web_visualizer_port,
            show_ui=cfg.rerun.open3d_web_show_ui,
        )

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
        frame_description=(
            "aruco"
            if aruco_alignment is not None and aruco_alignment.plane_frame is not None
            else raw.frame_description
        ),
        metadata={
            "backend": raw.backend_name,
            "run_name": run_name,
            "duration_s": time.time() - t0,
            **({"view_rejection": view_rejection_meta} if view_rejection_meta is not None else {}),
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
    aligned_markers_output_frame: Dict[int, np.ndarray] | None = None,
    aruco_geometric_qc: Dict[str, Any] | None = None,
    view_rejection_meta: Dict[str, Any] | None = None,
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
        sim3_to_save = (
            reflect_z_sim3_matrix(aruco_alignment.sim3)
            if cfg.output.z_axis_points_down
            else aruco_alignment.sim3.astype(np.float32)
        )
        np.save(run_dir / "sim3_to_output_frame.npy", sim3_to_save)
        _write_aruco_markers_3d(
            aruco_alignment,
            edge_length_m=cfg.aruco.marker_edge_length_m,
            path=run_dir / "aruco_markers_3d.json",
            corners_3d_output_frame=aligned_markers_output_frame,
        )
        _write_ba_diagnostics(
            aruco_alignment,
            path=run_dir / "aruco_bundle_adjustment.json",
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
        "z_axis_points_down": bool(cfg.output.z_axis_points_down),
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
        ba: BundleAdjustmentResult | None = aruco_alignment.bundle_adjustment
        if ba is not None:
            metadata["aruco_alignment"]["bundle_adjustment"] = {
                "scale_m_per_backend": float(ba.scale_m_per_backend),
                "initial_reproj_rmse_px": float(ba.initial_reproj_rmse_px),
                "final_reproj_rmse_px": float(ba.final_reproj_rmse_px),
                "num_observations": int(ba.num_observations),
                "num_iters": int(ba.num_iters),
                "converged": bool(ba.converged),
            }
        if aruco_geometric_qc is not None:
            metadata["aruco_alignment"]["geometric_qc"] = {
                k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in aruco_geometric_qc.items()
            }
    if view_rejection_meta is not None:
        metadata["view_rejection"] = view_rejection_meta
    (run_dir / "reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_ba_diagnostics(alignment: ArucoAlignment, path: Path) -> None:
    """Dump full BA residual breakdown (per-marker / per-view RMSE + scale)."""
    ba: BundleAdjustmentResult | None = alignment.bundle_adjustment
    if ba is None:
        return
    payload: Dict[str, Any] = {
        "scale_m_per_backend": float(ba.scale_m_per_backend),
        "initial_reproj_rmse_px": float(ba.initial_reproj_rmse_px),
        "final_reproj_rmse_px": float(ba.final_reproj_rmse_px),
        "num_observations": int(ba.num_observations),
        "num_iters": int(ba.num_iters),
        "converged": bool(ba.converged),
        "warnings": list(ba.warnings),
        "per_marker_reproj_rmse_px": {str(int(k)): float(v) for k, v in ba.per_marker_reproj_rmse_px.items()},
        "per_view_reproj_rmse_px": {str(int(k)): float(v) for k, v in ba.per_view_reproj_rmse_px.items()},
        "delta_T_per_view": ba.delta_T_per_view.tolist(),
        "marker_T_world": {str(int(k)): v.tolist() for k, v in ba.marker_T_world.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_to_rerun(
    run_dir: Path,
    raw,
    aligned_cloud: np.ndarray,
    aligned_colors: np.ndarray,
    aligned_confidence: np.ndarray,
    aligned_poses: np.ndarray,
    surface: SurfaceResult,
    aruco_alignment: ArucoAlignment | None,
    aligned_markers_in_output_frame: Dict[int, np.ndarray] | None,
    annotated_rgb_per_view: List[np.ndarray],
    confidence_percentile: float,
    rerun_cfg: RerunConfig,
) -> None:
    cloud_pts_vis, cloud_rgb_vis, cloud_conf_vis = resample_cloud_for_logging(
        points=aligned_cloud,
        colors=aligned_colors,
        confidence=aligned_confidence,
        factor=rerun_cfg.pointcloud_resample_factor,
    )

    if rerun_cfg.log_cloud_rgb or rerun_cfg.log_cloud_confidence:
        conf_colors = (
            _colorize_scalar_field(cloud_conf_vis, percentile=confidence_percentile)
            if rerun_cfg.log_cloud_confidence
            else None
        )
        log_cloud(
            cloud_pts_vis,
            cloud_rgb_vis,
            conf_colors=conf_colors,
            log_rgb=rerun_cfg.log_cloud_rgb,
            log_confidence=rerun_cfg.log_cloud_confidence,
        )

    if rerun_cfg.log_cameras:
        for idx in range(aligned_poses.shape[0]):
            K_orig = raw.K_per_view_orig[idx]
            orig_wh = (int(raw.original_image_sizes[idx, 0]), int(raw.original_image_sizes[idx, 1]))

            image_aruco = None
            if rerun_cfg.log_camera_aruco_images and idx < len(annotated_rgb_per_view):
                image_aruco = annotated_rgb_per_view[idx]

            conf_rgb = (
                _colorize_scalar_field(raw.confidence_maps_network[idx], percentile=confidence_percentile)
                if rerun_cfg.log_camera_confidence_images
                else None
            )

            log_camera(
                index=idx,
                T_world_cam=aligned_poses[idx],
                K=K_orig,
                resolution_wh=orig_wh,
                image_rgb=None,
                image_aruco_rgb=image_aruco,
                confidence_rgb=conf_rgb,
            )

    if rerun_cfg.log_aruco_3d and aruco_alignment is not None and aligned_markers_in_output_frame:
        log_aruco_alignment_in_output_frame(aligned_markers_in_output_frame)
    elif rerun_cfg.log_aruco_3d and not (aligned_markers_in_output_frame):
        print(
            "[yellow]Rerun: `log_aruco_3d` is enabled but there are no aligned 3D ArUco markers. "
            "Set `aruco.align_to_aruco: true` and ensure markers are detected, or set `log_aruco_3d: false`.[/yellow]"
        )

    if rerun_cfg.log_surface_mesh:
        log_surface_mesh(
            vertices=surface.points,
            triangles=surface.triangles,
            vertex_colors=surface.colors,
            vertex_normals=surface.normals,
        )

    rbl_path: Path | None = None
    if rerun_cfg.save_rbl:
        rbl_path = (Path(run_dir) / rerun_cfg.rerun_subdir / rerun_cfg.rbl_basename).resolve()
        rbl_path.parent.mkdir(parents=True, exist_ok=True)
    send_blueprint(
        include_scene_view=bool(
            rerun_cfg.log_cloud_rgb or rerun_cfg.log_cameras or rerun_cfg.log_aruco_3d or rerun_cfg.log_surface_mesh
        ),
        include_confidence_view=rerun_cfg.log_cloud_confidence,
        include_camera_images_view=rerun_cfg.log_camera_aruco_images or rerun_cfg.log_camera_confidence_images,
        include_cameras=rerun_cfg.log_cameras,
        include_cloud=rerun_cfg.log_cloud_rgb,
        include_cloud_confidence=rerun_cfg.log_cloud_confidence,
        include_aruco=rerun_cfg.log_aruco_3d,
        include_surface=rerun_cfg.log_surface_mesh,
        application_id="Spectra_Reconstruction",
        rbl_path=rbl_path if rerun_cfg.save_rbl else None,
    )
    rerun_flush()
    if (rerun_cfg.save_rrd or rerun_cfg.save_rbl) and HAS_RERUN:
        rd = (Path(run_dir) / rerun_cfg.rerun_subdir).resolve()
        parts: List[str] = []
        if rerun_cfg.save_rrd:
            parts.append(str((rd / rerun_cfg.rrd_basename).resolve()))
        if rerun_cfg.save_rbl:
            parts.append(str((rd / rerun_cfg.rbl_basename).resolve()))
        if parts:
            print(f"[dim]Rerun saved:[/dim] {', '.join(parts)}")


__all__ = ["ReconstructionResult", "run_reconstruction"]
