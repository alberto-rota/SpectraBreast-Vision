"""MASt3R back-end: pair inference, pose refinement, dense fusion.

Lifted from ``mast3r_pipeline.py::main`` and restructured as a pure function
returning a `RawReconstruction` that the orchestrator can align and feed to
`spectra.surface.reconstruct_surface`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.io as tv_io

from ..config import Mast3rConfig, ReconstructionConfig
from .types import BackendInputs, RawReconstruction


@dataclass(frozen=True)
class ImageGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    network_height: int
    network_width: int
    scale_x: float
    scale_y: float
    crop_left: float
    crop_top: float

    def original_to_network_intrinsics(self, K_orig: np.ndarray) -> np.ndarray:
        A = np.array(
            [
                [self.scale_x, 0.0, -self.crop_left],
                [0.0, self.scale_y, -self.crop_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A @ K_orig.astype(np.float32)

    def network_to_original_intrinsics(self, K_net: np.ndarray) -> np.ndarray:
        """Invert `original_to_network_intrinsics`: pulls K back into original pixel coords."""
        A_inv = np.array(
            [
                [1.0 / self.scale_x, 0.0, self.crop_left / self.scale_x],
                [0.0, 1.0 / self.scale_y, self.crop_top / self.scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A_inv @ K_net.astype(np.float32)

    def network_to_original(self, uv_net: np.ndarray) -> np.ndarray:
        uv_orig = uv_net.astype(np.float32).copy()
        uv_orig[:, 0] = (uv_orig[:, 0] + self.crop_left) / self.scale_x
        uv_orig[:, 1] = (uv_orig[:, 1] + self.crop_top) / self.scale_y
        return uv_orig

    def network_grid_to_original(self) -> tuple[np.ndarray, np.ndarray]:
        uu, vv = np.meshgrid(
            np.arange(self.network_width, dtype=np.float32),
            np.arange(self.network_height, dtype=np.float32),
        )
        uu = (uu + self.crop_left) / self.scale_x
        vv = (vv + self.crop_top) / self.scale_y
        return uu, vv

    def warp_network_map_to_original(self, map_net: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self.resized_height, self.resized_width), dtype=np.float32)
        left = int(round(self.crop_left))
        top = int(round(self.crop_top))
        canvas[top : top + self.network_height, left : left + self.network_width] = map_net.astype(np.float32)
        return cv2.resize(
            canvas,
            (self.original_width, self.original_height),
            interpolation=cv2.INTER_LINEAR,
        )


def _compute_image_geometry(image: torch.Tensor, image_size: int, patch_size: int, square_ok: bool = False) -> ImageGeometry:
    H0, W0 = int(image.shape[1]), int(image.shape[2])
    if image_size == 224:
        raise ValueError("MASt3R backend expects image_size != 224 so crop is trackable exactly.")

    long_edge = max(W0, H0)
    scale = image_size / float(long_edge)
    resized_width = int(round(W0 * scale))
    resized_height = int(round(H0 * scale))

    cx = resized_width // 2
    cy = resized_height // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2
    if (not square_ok) and resized_width == resized_height:
        halfh = 3 * halfw / 4

    crop_left = float(cx - halfw)
    crop_top = float(cy - halfh)
    network_width = int(round(2 * halfw))
    network_height = int(round(2 * halfh))

    return ImageGeometry(
        original_height=H0,
        original_width=W0,
        resized_height=resized_height,
        resized_width=resized_width,
        network_height=network_height,
        network_width=network_width,
        scale_x=resized_width / float(W0),
        scale_y=resized_height / float(H0),
        crop_left=crop_left,
        crop_top=crop_top,
    )


def _weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0:
        return points, colors, confidence

    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    weights = np.clip(confidence.astype(np.float64), 1e-6, None)
    weight_sum = np.bincount(inv, weights=weights, minlength=len(uniq))

    out_pts = np.zeros((len(uniq), 3), dtype=np.float64)
    out_cols = np.zeros((len(uniq), 3), dtype=np.float64)
    out_conf = np.bincount(inv, weights=confidence.astype(np.float64), minlength=len(uniq)) / np.maximum(counts, 1)

    for d in range(3):
        out_pts[:, d] = np.bincount(inv, weights=weights * points[:, d], minlength=len(uniq)) / np.maximum(
            weight_sum, 1e-12
        )
        out_cols[:, d] = np.bincount(inv, weights=weights * colors[:, d], minlength=len(uniq)) / np.maximum(
            weight_sum, 1e-12
        )

    return out_pts.astype(np.float32), np.clip(out_cols, 0, 255).astype(np.uint8), out_conf.astype(np.float32)


def _pairs_share_same_network_shape(pairs) -> bool:
    shapes1 = [tuple(pair[0]["img"].shape[-2:]) for pair in pairs]
    shapes2 = [tuple(pair[1]["img"].shape[-2:]) for pair in pairs]
    return all(shapes1[0] == shape for shape in shapes1) and all(shapes2[0] == shape for shape in shapes2)


def _make_temporal_pairs(images, neighbor_window: int):
    pairs = []
    num_images = len(images)
    for i in range(num_images):
        for j in range(i + 1, min(num_images, i + neighbor_window + 1)):
            pairs.append((images[i], images[j]))
    if num_images >= 2:
        last_idx = num_images - 1
        def _pair_first_idx(pair):
            idx = pair[0]["idx"]
            return int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])
        if not any(_pair_first_idx(p) == last_idx for p in pairs):
            pairs.append((images[last_idx], images[last_idx - 1]))
    return pairs


def _compact_pair_output(pair_output):
    return {
        "view1": {
            "img": pair_output["view1"]["img"],
            "true_shape": pair_output["view1"]["true_shape"],
            "idx": pair_output["view1"]["idx"],
            "instance": pair_output["view1"]["instance"],
        },
        "view2": {
            "img": pair_output["view2"]["img"],
            "true_shape": pair_output["view2"]["true_shape"],
            "idx": pair_output["view2"]["idx"],
            "instance": pair_output["view2"]["instance"],
        },
        "pred1": {"pts3d": pair_output["pred1"]["pts3d"], "conf": pair_output["pred1"]["conf"]},
        "pred2": {
            "pts3d_in_other_view": pair_output["pred2"]["pts3d_in_other_view"],
            "conf": pair_output["pred2"]["conf"],
        },
    }


@torch.no_grad()
def _match_pair_from_prediction(pred1, pred2, device, desc_conf_thr: float, max_matches_per_pair: int):
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs
    except Exception:
        from mast3r.utils.misc import fast_reciprocal_NNs

    desc1 = pred1["desc"]
    desc2 = pred2["desc"]
    desc_conf1 = pred1["desc_conf"]
    desc_conf2 = pred2["desc_conf"]

    if desc1.ndim == 4:
        desc1 = desc1.squeeze(0)
    if desc2.ndim == 4:
        desc2 = desc2.squeeze(0)
    if desc_conf1.ndim == 4:
        desc_conf1 = desc_conf1.squeeze(0)
    if desc_conf2.ndim == 4:
        desc_conf2 = desc_conf2.squeeze(0)
    if desc_conf1.ndim == 3:
        desc_conf1 = desc_conf1[..., 0]
    if desc_conf2.ndim == 3:
        desc_conf2 = desc_conf2[..., 0]

    valid1 = desc_conf1 >= desc_conf_thr
    valid2 = desc_conf2 >= desc_conf_thr
    y1, x1 = torch.where(valid1)
    y2, x2 = torch.where(valid2)
    if len(x1) == 0 or len(x2) == 0:
        return None, None, None

    xy1, xy2 = fast_reciprocal_NNs(
        desc1.float(),
        desc2.float(),
        subsample_or_initxy1=(x1.cpu().numpy(), y1.cpu().numpy()),
        ret_xy=True,
        device=device,
        dist="dot",
        block_size=2**13,
    )
    if xy1 is None or len(xy1) == 0:
        return None, None, None

    uv_i = xy1.astype(np.float32)
    uv_j = xy2.astype(np.float32)
    H1, W1 = desc_conf1.shape[0], desc_conf1.shape[1]
    H2, W2 = desc_conf2.shape[0], desc_conf2.shape[1]
    r1 = torch.as_tensor(xy1[:, 1], dtype=torch.long, device=desc_conf1.device).clamp(0, H1 - 1)
    c1 = torch.as_tensor(xy1[:, 0], dtype=torch.long, device=desc_conf1.device).clamp(0, W1 - 1)
    r2 = torch.as_tensor(xy2[:, 1], dtype=torch.long, device=desc_conf2.device).clamp(0, H2 - 1)
    c2 = torch.as_tensor(xy2[:, 0], dtype=torch.long, device=desc_conf2.device).clamp(0, W2 - 1)
    conf1_vals = desc_conf1.flatten(0, 1)[r1 * W1 + c1].cpu().numpy().astype(np.float32)
    conf2_vals = desc_conf2.flatten(0, 1)[r2 * W2 + c2].cpu().numpy().astype(np.float32)
    conf = np.minimum(conf1_vals, conf2_vals)

    if len(conf) > max_matches_per_pair:
        topk = np.argsort(conf)[-max_matches_per_pair:][::-1]
        uv_i = uv_i[topk]
        uv_j = uv_j[topk]
        conf = conf[topk]

    return uv_i, uv_j, conf


def _pixels_to_normalized_homogeneous(uv: np.ndarray, K: np.ndarray, device: torch.device) -> torch.Tensor:
    uv_t = torch.as_tensor(uv, dtype=torch.float32, device=device)
    x = torch.empty((uv_t.shape[0], 3), dtype=torch.float32, device=device)
    x[:, 0] = (uv_t[:, 0] - float(K[0, 2])) / float(K[0, 0])
    x[:, 1] = (uv_t[:, 1] - float(K[1, 2])) / float(K[1, 1])
    x[:, 2] = 1.0
    return x


def _skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vec[..., 0])
    x, y, z = vec.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    max_rad = np.pi
    eps = 1e-8
    norm_sq = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq + eps)
    scale = (norm.clamp(max=max_rad) / norm.clamp(min=eps)).clamp(max=1.0)
    rotvec_scaled = rotvec * scale
    theta2 = (rotvec_scaled * rotvec_scaled).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2 + eps)
    k = _skew_symmetric(rotvec_scaled)
    kk = k @ k
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0).repeat(rotvec.shape[0], 1, 1)
    A = torch.where(theta2 > eps, torch.sin(theta) / theta.clamp(min=eps), 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0)
    B = torch.where(theta2 > eps, (1.0 - torch.cos(theta)) / theta2.clamp(min=eps), 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0)
    return eye + A[..., None] * k + B[..., None] * kk


def _compose_refined_camera_poses(T_world_cam_init: torch.Tensor, rot_delta: torch.Tensor, trans_delta: torch.Tensor):
    num_cams = T_world_cam_init.shape[0]
    zero = torch.zeros((1, 3), dtype=T_world_cam_init.dtype, device=T_world_cam_init.device)
    if rot_delta.shape[0] == num_cams - 1:
        rot_full = torch.cat([zero, rot_delta], dim=0)
    else:
        rot_full = rot_delta
    if trans_delta.shape[0] == num_cams - 1:
        trans_full = torch.cat([zero, trans_delta], dim=0)
    else:
        trans_full = trans_delta

    R_init = T_world_cam_init[:, :3, :3]
    t_init = T_world_cam_init[:, :3, 3]
    R_delta = _axis_angle_to_matrix(rot_full)
    T_world_cam = torch.eye(4, dtype=T_world_cam_init.dtype, device=T_world_cam_init.device).unsqueeze(0).repeat(num_cams, 1, 1)
    T_world_cam[:, :3, :3] = R_init @ R_delta
    T_world_cam[:, :3, 3] = t_init + trans_full
    return T_world_cam, rot_full, trans_full


def _epipolar_error_px(T_world_cam, cam_i, cam_j, x1, x2, focal_px: float) -> torch.Tensor:
    R_world_cam = T_world_cam[:, :3, :3]
    t_world_cam = T_world_cam[:, :3, 3]
    R_i = R_world_cam[cam_i]
    R_j = R_world_cam[cam_j]
    t_i = t_world_cam[cam_i]
    t_j = t_world_cam[cam_j]

    R_ji = R_j.transpose(1, 2) @ R_i
    t_ji = (R_j.transpose(1, 2) @ (t_i - t_j).unsqueeze(-1)).squeeze(-1)
    E = _skew_symmetric(t_ji) @ R_ji

    Ex1 = (E @ x1.unsqueeze(-1)).squeeze(-1)
    Etx2 = (E.transpose(1, 2) @ x2.unsqueeze(-1)).squeeze(-1)
    x2tEx1 = (x2 * Ex1).sum(dim=-1)
    denom = Ex1[:, :2].square().sum(dim=-1) + Etx2[:, :2].square().sum(dim=-1) + 1e-6
    sampson = (x2tEx1.square() / denom).clamp(min=0.0)
    return torch.sqrt(sampson + 1e-6) * focal_px


def _prepare_pose_refinement_data(match_records, K: np.ndarray, device: torch.device):
    if len(match_records) == 0:
        return None
    cam_i_parts, cam_j_parts, pair_id_parts = [], [], []
    x1_parts, x2_parts, weight_parts = [], [], []
    pair_names, pair_match_counts = [], []

    for pair_id, rec in enumerate(match_records):
        x1 = _pixels_to_normalized_homogeneous(rec["uv_i"], K, device)
        x2 = _pixels_to_normalized_homogeneous(rec["uv_j"], K, device)
        weight = torch.as_tensor(rec["conf"], dtype=torch.float32, device=device).clamp_min(1e-4)
        weight = weight / weight.mean().clamp_min(1e-6)
        cam_i_parts.append(torch.full((x1.shape[0],), rec["i"], dtype=torch.long, device=device))
        cam_j_parts.append(torch.full((x1.shape[0],), rec["j"], dtype=torch.long, device=device))
        pair_id_parts.append(torch.full((x1.shape[0],), pair_id, dtype=torch.long, device=device))
        x1_parts.append(x1)
        x2_parts.append(x2)
        weight_parts.append(weight)
        pair_names.append(f"{rec['i']}-{rec['j']}")
        pair_match_counts.append(int(x1.shape[0]))

    return {
        "cam_i": torch.cat(cam_i_parts, dim=0),
        "cam_j": torch.cat(cam_j_parts, dim=0),
        "pair_id": torch.cat(pair_id_parts, dim=0),
        "x1": torch.cat(x1_parts, dim=0),
        "x2": torch.cat(x2_parts, dim=0),
        "weight": torch.cat(weight_parts, dim=0),
        "num_pairs": len(match_records),
        "num_matches": int(sum(pair_match_counts)),
        "pair_names": pair_names,
        "pair_match_counts": pair_match_counts,
    }


def _refine_camera_poses_with_epipolar(
    match_records,
    T_world_cam_init_np: np.ndarray,
    K_orig: np.ndarray,
    mast3r_cfg: Mast3rConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    zero_rot = np.zeros((len(T_world_cam_init_np), 3), dtype=np.float32)
    zero_trans = np.zeros((len(T_world_cam_init_np), 3), dtype=np.float32)
    pose_stats = {
        "pairs_used": len(match_records),
        "matches_used": int(sum(len(rec["conf"]) for rec in match_records)),
        "median_epipolar_px_before": None,
        "mean_epipolar_px_before": None,
        "median_epipolar_px_after": None,
        "mean_epipolar_px_after": None,
        "rotation_delta_axis_angle_rad": zero_rot.tolist(),
        "translation_delta_m": zero_trans.tolist(),
        "rotation_delta_deg_norm": np.linalg.norm(np.rad2deg(zero_rot), axis=1).tolist(),
        "translation_delta_m_norm": np.linalg.norm(zero_trans, axis=1).tolist(),
        "pair_match_counts": [],
        "used_alignment": False,
    }

    if len(T_world_cam_init_np) < 2 or len(match_records) == 0 or mast3r_cfg.pose_refine_iters <= 0:
        return T_world_cam_init_np.astype(np.float32), pose_stats

    problem = _prepare_pose_refinement_data(match_records, K_orig, device)
    pose_stats["pair_match_counts"] = [
        {"pair": pair_name, "matches": count}
        for pair_name, count in zip(problem["pair_names"], problem["pair_match_counts"])
    ]

    focal_px = float(np.sqrt(float(K_orig[0, 0]) * float(K_orig[1, 1])))
    T_world_cam_init = torch.as_tensor(T_world_cam_init_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        err_before = _epipolar_error_px(
            T_world_cam_init, problem["cam_i"], problem["cam_j"], problem["x1"], problem["x2"], focal_px=focal_px
        )
        pose_stats["median_epipolar_px_before"] = float(err_before.median().item())
        pose_stats["mean_epipolar_px_before"] = float(err_before.mean().item())

    rot_delta = torch.zeros((len(T_world_cam_init_np) - 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros((len(T_world_cam_init_np) - 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rot_delta, trans_delta], lr=max(mast3r_cfg.pose_refine_lr, 0.0))

    best_median_px = float("inf")
    best_rot = rot_delta.detach().clone()
    best_trans = trans_delta.detach().clone()
    huber_beta = max(float(mast3r_cfg.pixel_tol), 1e-4)
    sigma_rot_deg = max(float(mast3r_cfg.pose_prior_sigma_deg), 1e-6)
    sigma_trans_m = max(float(mast3r_cfg.pose_prior_sigma_m), 1e-6)
    max_rot_rad = np.pi
    max_trans_m = 0.2
    grad_clip_norm = 1.0

    def _cosine_lr(step, total, lr_max, lr_min):
        t = step / max(total - 1, 1)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t))

    for iteration in range(mast3r_cfg.pose_refine_iters):
        lr = _cosine_lr(iteration, mast3r_cfg.pose_refine_iters, float(mast3r_cfg.pose_refine_lr), float(mast3r_cfg.pose_refine_lr_min))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        T_world_cam_refined, rot_full, trans_full = _compose_refined_camera_poses(T_world_cam_init, rot_delta, trans_delta)
        err_px = _epipolar_error_px(T_world_cam_refined, problem["cam_i"], problem["cam_j"], problem["x1"], problem["x2"], focal_px=focal_px)
        robust_err = torch.where(err_px <= huber_beta, 0.5 * err_px.square() / huber_beta, err_px - 0.5 * huber_beta)

        pair_loss = torch.zeros((problem["num_pairs"],), dtype=torch.float32, device=device)
        pair_weight = torch.zeros((problem["num_pairs"],), dtype=torch.float32, device=device)
        pair_loss.index_add_(0, problem["pair_id"], robust_err * problem["weight"])
        pair_weight.index_add_(0, problem["pair_id"], problem["weight"])
        loss_epipolar = (pair_loss / pair_weight.clamp_min(1e-6)).mean()

        rot_norm_deg = torch.rad2deg(rot_delta.norm(dim=-1))
        trans_norm_m = trans_delta.norm(dim=-1)
        loss_prior = ((rot_norm_deg / sigma_rot_deg).square() + (trans_norm_m / sigma_trans_m).square()).mean()
        loss = loss_epipolar + float(mast3r_cfg.pose_prior_weight) * loss_prior

        median_px_iter = float(err_px.median().item()) if err_px.isfinite().any() else float("nan")
        max_drot_iter = float(rot_norm_deg.max().item()) if rot_norm_deg.isfinite().any() else float("inf")
        max_dt_iter = float(trans_norm_m.max().item()) if trans_norm_m.isfinite().any() else float("inf")
        within_cap = (
            max_drot_iter <= float(mast3r_cfg.pose_refine_max_drot_deg)
            and max_dt_iter <= float(mast3r_cfg.pose_refine_max_dt_m)
        )
        if np.isfinite(median_px_iter) and median_px_iter < best_median_px and within_cap:
            best_median_px = median_px_iter
            best_rot = rot_delta.detach().clone()
            best_trans = trans_delta.detach().clone()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([rot_delta, trans_delta], max_norm=grad_clip_norm)
        optimizer.step()
        with torch.no_grad():
            rot_norm = rot_delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            rot_delta.data.mul_(rot_norm.clamp(max=max_rot_rad) / rot_norm)
            trans_delta.data.clamp_(-max_trans_m, max_trans_m)
            rot_delta.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            trans_delta.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        T_world_cam_best, rot_full_best, trans_full_best = _compose_refined_camera_poses(T_world_cam_init, best_rot, best_trans)
        err_after = _epipolar_error_px(T_world_cam_best, problem["cam_i"], problem["cam_j"], problem["x1"], problem["x2"], focal_px=focal_px)

    rot_full_np = rot_full_best.detach().cpu().numpy().astype(np.float32)
    trans_full_np = trans_full_best.detach().cpu().numpy().astype(np.float32)
    pose_stats.update(
        {
            "median_epipolar_px_after": float(err_after.median().item()),
            "mean_epipolar_px_after": float(err_after.mean().item()),
            "rotation_delta_axis_angle_rad": rot_full_np.tolist(),
            "translation_delta_m": trans_full_np.tolist(),
            "rotation_delta_deg_norm": np.linalg.norm(np.rad2deg(rot_full_np), axis=1).tolist(),
            "translation_delta_m_norm": np.linalg.norm(trans_full_np, axis=1).tolist(),
            "used_alignment": True,
        }
    )
    return T_world_cam_best.detach().cpu().numpy().astype(np.float32), pose_stats


def _patch_modular_optimizer():
    from dust3r.cloud_opt.modular_optimizer import ModularPointCloudOptimizer

    def get_known_focal_mask(self):
        return torch.tensor([not focal.requires_grad for focal in self.im_focals], device=self.device)

    ModularPointCloudOptimizer.get_known_focal_mask = get_known_focal_mask
    return ModularPointCloudOptimizer


@torch.no_grad()
def _preset_exact_intrinsics(scene, intrinsics_network):
    for idx, K in enumerate(intrinsics_network):
        focal = torch.tensor([K[0, 0], K[1, 1]], dtype=torch.float32, device=scene.device)
        pp = torch.tensor(K[:2, 2], dtype=torch.float32, device=scene.device)
        scene._no_grad(scene._set_focal(idx, focal, force=True))
        scene._no_grad(scene._set_principal_point(idx, pp, force=True))


def _ensure_mast3r_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    mast3r_root = repo_root / "mast3r"
    if str(mast3r_root) not in sys.path:
        sys.path.append(str(mast3r_root))


def run_mast3r(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    """Run MASt3R pair inference + global alignment + dense fusion."""
    _ensure_mast3r_on_path()
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.inference import inference
    from dust3r.utils.device import collate_with_cat
    from dust3r.utils.image import load_images
    from mast3r.model import AsymmetricMASt3R

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = inputs.image_paths
    num_views = len(image_paths)
    if num_views < 2:
        raise RuntimeError("MASt3R backend needs at least two images.")

    have_gt_poses = inputs.T_world_cam_gt is not None and inputs.T_world_cam_gt.shape[0] == num_views
    have_gt_intrinsics = inputs.K_orig_gt is not None
    K_orig = inputs.K_orig_gt

    mcfg: Mast3rConfig = cfg.mast3r

    orig_images = [tv_io.read_image(str(path)) for path in image_paths]  # [3, H, W] uint8

    model = AsymmetricMASt3R.from_pretrained(mcfg.model_name).to(device)
    model.eval()
    patch_size = int(getattr(model, "patch_size", 16))

    filelist = [str(path) for path in image_paths]
    imgs_mast3r = load_images(filelist, size=mcfg.image_size, verbose=True, patch_size=patch_size)
    geometries = [
        _compute_image_geometry(image, image_size=mcfg.image_size, patch_size=patch_size, square_ok=False)
        for image in orig_images
    ]
    K_network = (
        [geom.original_to_network_intrinsics(K_orig) for geom in geometries]
        if have_gt_intrinsics
        else None
    )

    for image_idx, (geom, img_net) in enumerate(zip(geometries, imgs_mast3r)):
        net_h, net_w = map(int, img_net["img"].shape[-2:])
        if (geom.network_height, geom.network_width) != (net_h, net_w):
            raise RuntimeError(
                f"Network geometry mismatch for image {image_idx}: expected "
                f"{(geom.network_height, geom.network_width)}, got {(net_h, net_w)}"
            )

    pairs = _make_temporal_pairs(imgs_mast3r, mcfg.neighbor_window)
    if len(pairs) == 0:
        raise RuntimeError("No temporal pairs were generated. Increase --mast3r.neighbor_window.")

    compact_pair_results: List[dict] = []
    match_records: List[dict] = []

    for pair_idx, (img_i, img_j) in enumerate(pairs):
        image_i = int(img_i["idx"])
        image_j = int(img_j["idx"])
        pair_output = inference([(img_i, img_j)], model, device, batch_size=1, verbose=False)

        uv_i_net, uv_j_net, match_conf = _match_pair_from_prediction(
            pair_output["pred1"],
            pair_output["pred2"],
            device=device,
            desc_conf_thr=mcfg.desc_conf_thr,
            max_matches_per_pair=mcfg.max_matches_per_pair,
        )
        if uv_i_net is not None and len(uv_i_net) > 0:
            uv_i_orig = geometries[image_i].network_to_original(uv_i_net)
            uv_j_orig = geometries[image_j].network_to_original(uv_j_net)
            match_records.append(
                {"i": image_i, "j": image_j, "uv_i": uv_i_orig, "uv_j": uv_j_orig, "conf": match_conf}
            )
        compact_pair_results.append(_compact_pair_output(pair_output))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    multiple_shapes = not _pairs_share_same_network_shape(pairs)
    compact_output = collate_with_cat(compact_pair_results, lists=multiple_shapes)

    if have_gt_poses and have_gt_intrinsics:
        refined_poses, pose_stats = _refine_camera_poses_with_epipolar(
            match_records=match_records,
            T_world_cam_init_np=inputs.T_world_cam_gt,
            K_orig=K_orig,
            mast3r_cfg=mcfg,
            device=device,
        )
    else:
        refined_poses = inputs.T_world_cam_gt if have_gt_poses else None
        pose_stats = {
            "pairs_used": len(match_records),
            "matches_used": int(sum(len(rec["conf"]) for rec in match_records)),
            "used_alignment": False,
            "note": "Skipped epipolar refinement (RGB-only or missing intrinsics)",
        }

    _patch_modular_optimizer()
    scene = global_aligner(
        compact_output,
        device=device,
        mode=GlobalAlignerMode.ModularPointCloudOptimizer,
        verbose=True,
        min_conf_thr=mcfg.dense_conf_thr,
        fx_and_fy=bool(have_gt_intrinsics),
    )

    if have_gt_poses:
        refined_poses_t = torch.tensor(refined_poses, dtype=torch.float32, device=device)
        scene.preset_pose(refined_poses_t)
    if have_gt_intrinsics:
        _preset_exact_intrinsics(scene.cpu(), K_network)

    dense_init = "known_poses" if have_gt_poses else "mst"
    scene.compute_global_alignment(
        init=dense_init,
        niter=mcfg.dense_refine_iters,
        schedule="cosine",
        lr=mcfg.dense_refine_lr,
    )

    with torch.no_grad():
        refined_poses = scene.get_im_poses().detach().cpu().numpy().astype(np.float32)
        scene_intrinsics_network = scene.get_intrinsics().detach().cpu().numpy().astype(np.float32)

    if K_network is None:
        K_network = [K for K in scene_intrinsics_network]

    scene = scene.clean_pointcloud()
    dense_conf_thr_raw = float(scene.conf_trf(torch.tensor(mcfg.dense_conf_thr)))
    scene.min_conf_thr = dense_conf_thr_raw

    with torch.no_grad():
        pts3d_list = [pts.detach().cpu().numpy() for pts in scene.get_pts3d()]   # [V] of [Hn,Wn,3]
        conf_list = [conf.detach().cpu().numpy() for conf in scene.im_conf]      # [V] of [Hn,Wn]
        mask_list = [mask.detach().cpu().numpy() for mask in scene.get_masks()]  # [V] of [Hn,Wn]

    all_pts_list: List[np.ndarray] = []
    all_cols_list: List[np.ndarray] = []
    all_conf_list: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    point_map_world: List[np.ndarray] = []
    images_network_uint8: List[np.ndarray] = []
    confidence_maps_network: List[np.ndarray] = []

    for image_idx, (pts3d, conf_map, mask, geom, image) in enumerate(
        zip(pts3d_list, conf_list, mask_list, geometries, orig_images)
    ):
        rgb_orig = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        uu_orig, vv_orig = geom.network_grid_to_original()
        u_idx = np.clip(np.round(uu_orig).astype(np.int32), 0, geom.original_width - 1)
        v_idx = np.clip(np.round(vv_orig).astype(np.int32), 0, geom.original_height - 1)
        cols_full = rgb_orig[v_idx, u_idx].astype(np.uint8)

        valid = mask & np.isfinite(pts3d).all(axis=-1)
        if not np.any(valid):
            valid = (conf_map > 0) & np.isfinite(pts3d).all(axis=-1)

        pts = pts3d[valid].astype(np.float32)
        cols = cols_full[valid].astype(np.uint8)
        conf = conf_map[valid].astype(np.float32)
        all_pts_list.append(pts)
        all_cols_list.append(cols)
        all_conf_list.append(conf)
        valid_masks.append(valid)
        point_map_world.append(pts3d.astype(np.float32))
        images_network_uint8.append(cols_full)
        confidence_maps_network.append(conf_map.astype(np.float32))

    if not any(len(p) > 0 for p in all_pts_list):
        raise RuntimeError("MASt3R fusion produced no valid points.")

    fused_points = np.concatenate(all_pts_list, axis=0)
    fused_colors = np.concatenate(all_cols_list, axis=0)
    fused_confidence = np.concatenate(all_conf_list, axis=0)

    if mcfg.voxel_size > 0.0:
        fused_points, fused_colors, fused_confidence = _weighted_voxel_downsample(
            fused_points, fused_colors, fused_confidence, mcfg.voxel_size
        )

    if len(fused_points) > mcfg.max_points:
        keep = np.argsort(fused_confidence)[-mcfg.max_points:][::-1]
        fused_points = fused_points[keep]
        fused_colors = fused_colors[keep]
        fused_confidence = fused_confidence[keep]

    valid_masks_np = np.stack(valid_masks, axis=0)
    point_map_world_np = np.stack(point_map_world, axis=0)
    images_net_np = np.stack(images_network_uint8, axis=0)
    conf_maps_np = np.stack(confidence_maps_network, axis=0)
    K_net_per_view = np.stack(K_network, axis=0).astype(np.float32)

    if have_gt_intrinsics:
        K_orig_per_view = np.repeat(K_orig[None].astype(np.float32), num_views, axis=0)
    else:
        K_orig_per_view = np.stack(
            [geom.network_to_original_intrinsics(K_net_per_view[i]) for i, geom in enumerate(geometries)],
            axis=0,
        ).astype(np.float32)

    network_image_sizes = np.asarray(
        [[g.network_width, g.network_height] for g in geometries], dtype=np.int32
    )
    original_image_sizes = np.asarray(
        [[g.original_width, g.original_height] for g in geometries], dtype=np.int32
    )

    return RawReconstruction(
        fused_points=fused_points,
        fused_colors=fused_colors,
        fused_confidence=fused_confidence,
        point_map_world=point_map_world_np,
        valid_masks=valid_masks_np,
        T_world_cam=refined_poses.astype(np.float32),
        K_per_view_orig=K_orig_per_view,
        K_per_view_network=K_net_per_view,
        network_image_sizes=network_image_sizes,
        original_image_sizes=original_image_sizes,
        images_network_uint8=images_net_np,
        confidence_maps_network=conf_maps_np,
        frame_description=("gt-aligned" if have_gt_poses else "mst"),
        backend_name="mast3r",
        alignment_info={"pose_refinement": pose_stats},
        extra={
            "have_gt_poses": bool(have_gt_poses),
            "have_gt_intrinsics": bool(have_gt_intrinsics),
            "num_pairs": len(pairs),
            "num_match_records": len(match_records),
            "dense_init": dense_init,
            "dense_conf_thr_raw": dense_conf_thr_raw,
            "geometries": geometries,
        },
    )


__all__ = ["ImageGeometry", "run_mast3r"]
