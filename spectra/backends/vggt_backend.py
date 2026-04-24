"""VGGT back-end: feed-forward inference + fusion into `RawReconstruction`.

This is the functional lift of ``vggt_pipeline.py::main`` (steps 1-6): it
runs VGGT, optionally aligns the predicted frame to provided GT poses, and
returns a back-end-agnostic `RawReconstruction` that the orchestrator can
further align to ArUco markers and feed into surface reconstruction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.io as tv_io

from ..config import ReconstructionConfig
from .types import BackendInputs, RawReconstruction


def _camera_from_world_extrinsics_to_world_from_camera(T_cam_world_3x4: np.ndarray) -> np.ndarray:
    """Convert OpenCV [R|t] world->cam extrinsics into 4x4 cam->world poses.

    Input:  [S, 3, 4]
    Output: [S, 4, 4]
    """
    T = np.asarray(T_cam_world_3x4, dtype=np.float32)
    S = T.shape[0]
    T_cw = np.repeat(np.eye(4, dtype=np.float32)[None], S, axis=0)
    T_cw[:, :3, :4] = T
    T_wc = np.linalg.inv(T_cw).astype(np.float32)
    return T_wc


def _fit_umeyama_alignment(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    estimate_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment: ``dst ≈ s * R @ src + t``."""
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src,dst shape [N,3], got {src.shape} vs {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for a stable similarity transform")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1.0
    R = U @ np.diag(sign) @ Vt

    if estimate_scale:
        var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
        if var_src < 1e-12:
            raise ValueError("Degenerate source configuration")
        scale = float(np.sum(D * sign) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R.astype(np.float32), t.astype(np.float32)


def _apply_similarity_transform(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape
    pts_flat = pts.reshape(-1, 3).astype(np.float64)
    out = float(scale) * (pts_flat @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64)[None, :]
    return out.astype(np.float32).reshape(orig_shape)


def _apply_similarity_to_camera_poses(
    T_world_cam: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> np.ndarray:
    T = np.asarray(T_world_cam, dtype=np.float32)
    out = np.repeat(np.eye(4, dtype=np.float32)[None], T.shape[0], axis=0)
    R64 = np.asarray(R_align, dtype=np.float64)
    t64 = np.asarray(t_align, dtype=np.float64)
    out[:, :3, :3] = np.einsum("ij,vjk->vik", R64, T[:, :3, :3].astype(np.float64)).astype(np.float32)
    out[:, :3, 3] = (
        float(scale) * np.einsum("ij,vj->vi", R64, T[:, :3, 3].astype(np.float64)) + t64[None, :]
    ).astype(np.float32)
    return out


def _compute_image_geometries(
    image_paths: List[Path],
    target_size: int,
    K_orig_shared: np.ndarray | None,
) -> List[dict]:
    """Match VGGT's ``load_and_preprocess_images(..., mode='crop')`` geometry.

    Returns per-image dict with original size, network size, scale factors,
    and optionally ``K_net`` (if GT intrinsics were provided).
    """
    geoms: List[dict] = []
    for path in image_paths:
        img = tv_io.read_image(str(path))  # [3, H, W] uint8
        H = int(img.shape[1])
        W = int(img.shape[2])
        new_width = int(target_size)
        new_height = int(round(H * (new_width / W) / 14.0) * 14)
        scale_x = new_width / W
        scale_y = new_height / H
        start_y = (new_height - target_size) // 2 if new_height > target_size else 0
        crop_h = target_size if new_height > target_size else new_height

        K_net: np.ndarray | None = None
        if K_orig_shared is not None:
            K_net = np.eye(3, dtype=np.float32)
            K_net[0, 0] = float(K_orig_shared[0, 0]) * scale_x
            K_net[1, 1] = float(K_orig_shared[1, 1]) * scale_y
            K_net[0, 2] = float(K_orig_shared[0, 2]) * scale_x
            K_net[1, 2] = float(K_orig_shared[1, 2]) * scale_y - start_y

        geoms.append(
            {
                "original_width": W,
                "original_height": H,
                "new_width": new_width,
                "new_height": new_height,
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "start_y": int(start_y),
                "crop_h": int(crop_h),
                "K_net": K_net,
            }
        )
    return geoms


def _k_net_to_k_orig(K_net: np.ndarray, geom: dict) -> np.ndarray:
    """Convert a network-pixel intrinsics matrix back to original-pixel coords.

    This is used after inference when GT intrinsics were not provided so that
    ArUco triangulation (which uses original-resolution detections) still
    has a valid K per view.
    """
    scale_x = float(geom["scale_x"])
    scale_y = float(geom["scale_y"])
    start_y = float(geom["start_y"])
    K_out = np.eye(3, dtype=np.float32)
    K_out[0, 0] = float(K_net[0, 0]) / scale_x
    K_out[1, 1] = float(K_net[1, 1]) / scale_y
    K_out[0, 2] = float(K_net[0, 2]) / scale_x
    K_out[1, 2] = (float(K_net[1, 2]) + start_y) / scale_y
    return K_out


def _ensure_vggt_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    vggt_root = repo_root / "vggt"
    if str(vggt_root) not in sys.path:
        sys.path.append(str(vggt_root))


def run_vggt(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    """Run VGGT inference and fuse points into a backend-agnostic reconstruction."""
    _ensure_vggt_on_path()
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    image_paths = inputs.image_paths
    num_views = len(image_paths)
    have_gt_poses = inputs.T_world_cam_gt is not None and inputs.T_world_cam_gt.shape[0] == num_views
    have_gt_intrinsics = inputs.K_orig_gt is not None

    vggt_cfg = cfg.vggt
    camera_source = vggt_cfg.camera_source
    if camera_source == "gt" and not have_gt_poses:
        camera_source = "predicted"

    geometries = _compute_image_geometries(image_paths, target_size=vggt_cfg.image_size, K_orig_shared=inputs.K_orig_gt)

    model = VGGT.from_pretrained(vggt_cfg.model_name).to(device=device, dtype=amp_dtype)
    model.eval()

    images_vggt = load_and_preprocess_images([str(p) for p in image_paths], mode="crop").to(device)  # [S, 3, Hn, Wn]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            batch = images_vggt.unsqueeze(0)  # [1, S, 3, Hn, Wn]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic_pred, intrinsic_pred = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])

            if vggt_cfg.cloud_source == "depth_map":
                depth_map, map_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)
                point_map = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    extrinsic_pred.squeeze(0),
                    intrinsic_pred.squeeze(0),
                )  # numpy [S, Hn, Wn, 3]
            else:
                point_map_pred, map_conf = model.point_head(aggregated_tokens_list, batch, ps_idx)
                point_map = point_map_pred.squeeze(0)[..., :3].float().cpu().numpy()

            map_conf_np = map_conf.squeeze(0).float().cpu().numpy()                        # [S, Hn, Wn]
            intrinsic_pred_np = intrinsic_pred.squeeze(0).float().cpu().numpy()            # [S, 3, 3]
            extrinsic_pred_np = extrinsic_pred.squeeze(0).float().cpu().numpy()            # [S, 3, 4]

    del model, batch, aggregated_tokens_list, pose_enc, extrinsic_pred, intrinsic_pred
    if device.type == "cuda":
        torch.cuda.empty_cache()

    T_world_cam_pred = _camera_from_world_extrinsics_to_world_from_camera(extrinsic_pred_np)  # [S,4,4]
    T_world_cam_reconstruction = T_world_cam_pred.copy()
    point_map_in_output_frame = point_map.copy()

    alignment_info = None
    if camera_source == "gt":
        pred_centers = T_world_cam_pred[:, :3, 3]
        gt_centers = inputs.T_world_cam_gt[:, :3, 3]
        estimate_scale = vggt_cfg.alignment_mode == "sim3"
        scale_align, R_align, t_align = _fit_umeyama_alignment(pred_centers, gt_centers, estimate_scale=estimate_scale)

        pred_centers_aligned = _apply_similarity_transform(pred_centers, scale_align, R_align, t_align)
        rmse_before = float(np.sqrt(np.mean(np.sum((pred_centers - gt_centers) ** 2, axis=1))))
        rmse_after = float(np.sqrt(np.mean(np.sum((pred_centers_aligned - gt_centers) ** 2, axis=1))))

        point_map_in_output_frame = _apply_similarity_transform(point_map, scale_align, R_align, t_align)
        T_world_cam_reconstruction = _apply_similarity_to_camera_poses(T_world_cam_pred, scale_align, R_align, t_align)

        alignment_info = {
            "mode": vggt_cfg.alignment_mode,
            "scale": float(scale_align),
            "rotation_matrix": R_align.tolist(),
            "translation": t_align.tolist(),
            "camera_center_rmse_before_m": rmse_before,
            "camera_center_rmse_after_m": rmse_after,
        }

    if not have_gt_intrinsics:
        for idx, geom in enumerate(geometries):
            geom["K_net"] = intrinsic_pred_np[idx].astype(np.float32)

    K_net_per_view = np.stack([g["K_net"] for g in geometries], axis=0).astype(np.float32)  # [S,3,3]
    if have_gt_intrinsics:
        K_orig_per_view = np.repeat(inputs.K_orig_gt[None].astype(np.float32), num_views, axis=0)
    else:
        K_orig_per_view = np.stack(
            [_k_net_to_k_orig(geom["K_net"], geom) for geom in geometries], axis=0
        ).astype(np.float32)

    images_np = images_vggt.permute(0, 2, 3, 1).cpu().numpy()  # [S, Hn, Wn, 3] in [0,1]
    images_u8 = np.clip(images_np * 255.0, 0.0, 255.0).astype(np.uint8)

    valid_masks: List[np.ndarray] = []
    all_pts_list: List[np.ndarray] = []
    all_cols_list: List[np.ndarray] = []
    all_conf_list: List[np.ndarray] = []

    for img_idx in range(num_views):
        P_world = point_map_in_output_frame[img_idx]  # [Hn, Wn, 3]
        conf = map_conf_np[img_idx]
        cols = images_u8[img_idx]

        if vggt_cfg.conf_thres == 0.0:
            conf_thr = 0.0
        else:
            conf_thr = float(np.percentile(conf, vggt_cfg.conf_thres))
        conf_mask = (conf >= conf_thr) & (conf > 1e-5)

        if vggt_cfg.mask_black_bg:
            conf_mask &= cols.sum(axis=2) >= 16
        if vggt_cfg.mask_white_bg:
            conf_mask &= ~(
                (cols[:, :, 0] > 240) & (cols[:, :, 1] > 240) & (cols[:, :, 2] > 240)
            )

        finite = np.isfinite(P_world).all(axis=-1) & (np.linalg.norm(P_world, axis=-1) > 1e-6)
        valid = conf_mask & finite
        valid_masks.append(valid)

        if np.any(valid):
            all_pts_list.append(P_world[valid].astype(np.float32))
            all_cols_list.append(cols[valid].astype(np.uint8))
            all_conf_list.append(conf[valid].astype(np.float32))

    if not all_pts_list:
        raise RuntimeError("VGGT produced no valid points after filtering; try lowering --vggt.conf_thres.")

    fused_points = np.concatenate(all_pts_list, axis=0)
    fused_colors = np.concatenate(all_cols_list, axis=0)
    fused_confidence = np.concatenate(all_conf_list, axis=0)
    valid_masks_np = np.stack(valid_masks, axis=0)

    network_image_sizes = np.asarray(
        [[g["new_width"], g["crop_h"]] for g in geometries],
        dtype=np.int32,
    )
    original_image_sizes = np.asarray(
        [[g["original_width"], g["original_height"]] for g in geometries],
        dtype=np.int32,
    )

    return RawReconstruction(
        fused_points=fused_points,
        fused_colors=fused_colors,
        fused_confidence=fused_confidence,
        point_map_world=point_map_in_output_frame.astype(np.float32),
        valid_masks=valid_masks_np,
        T_world_cam=T_world_cam_reconstruction.astype(np.float32),
        K_per_view_orig=K_orig_per_view,
        K_per_view_network=K_net_per_view,
        network_image_sizes=network_image_sizes,
        original_image_sizes=original_image_sizes,
        images_network_uint8=images_u8,
        confidence_maps_network=map_conf_np.astype(np.float32),
        frame_description=("gt-aligned" if camera_source == "gt" else "predicted"),
        backend_name="vggt",
        alignment_info=alignment_info,
        extra={
            "have_gt_poses": bool(have_gt_poses),
            "have_gt_intrinsics": bool(have_gt_intrinsics),
            "T_world_cam_predicted": T_world_cam_pred.astype(np.float32),
            "intrinsics_predicted": intrinsic_pred_np.astype(np.float32),
            "geometries": geometries,
        },
    )


__all__ = ["run_vggt"]
