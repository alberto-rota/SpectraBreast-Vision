import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
from rich import print

import rerun as rr
from helpers import xyzeuler_to_hmat

import rerun.blueprint as rrb


def _send_rerun_blueprint() -> None:
    """Send a tidy Rerun blueprint: Scene (points + cameras), Confidence 3D, and camera images."""
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/",
                    contents=["/points", "/cameras"],
                ),
                rrb.Spatial3DView(
                    name="Confidence",
                    origin="/",
                    contents=["/points_confidence"],
                ),
            ),
            rrb.Spatial2DView(
                name="Camera images",
                origin="/cameras",
            ),
        ),
    )
    rr.send_blueprint(blueprint, make_active=True)



def _parse_args():
    p = argparse.ArgumentParser(
        description="Metric dense MASt3R fusion with pose refinement, exact calibration, and confidence logging"
    )
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction"))

    # MASt3R
    p.add_argument(
        "--model_name",
        type=str,
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    )
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument(
        "--neighbor_window",
        type=int,
        default=2,
        help="Use temporal neighbors i+1...i+k for matching/fusion",
    )

    # Matching / alignment
    p.add_argument(
        "--desc_conf_thr",
        type=float,
        default=0.1,
        help="Descriptor confidence threshold; lower = more matches, more outliers",
    )
    p.add_argument(
        "--pixel_tol",
        type=float,
        default=1.5,
        help="Robust epipolar loss transition point in pixels",
    )
    p.add_argument("--max_matches_per_pair", type=int, default=50000)
    p.add_argument(
        "--min_reproj_px",
        type=float,
        default=2.0,
        help="Reserved for backward compatibility with the previous script",
    )
    p.add_argument(
        "--min_triang_angle_deg",
        type=float,
        default=1.0,
        help="Reserved for backward compatibility with the previous script",
    )
    p.add_argument(
        "--pose_refine_iters",
        type=int,
        default=1,
        help="Pose refinement iterations; avoid very high values to prevent overfitting",
    )
    p.add_argument(
        "--pose_refine_lr",
        type=float,
        default=0,
        help="Initial learning rate for pose refinement",
    )
    p.add_argument(
        "--pose_refine_lr_min",
        type=float,
        default=1e-5,
        help="Minimum LR for cosine decay during pose refinement",
    )
    p.add_argument("--pose_prior_sigma_deg", type=float, default=1.0)
    p.add_argument("--pose_prior_sigma_m", type=float, default=0.01)
    p.add_argument(
        "--pose_prior_weight",
        type=float,
        default=0.06,
        help="Weight of pose prior; lower allows more refinement from matches",
    )
    p.add_argument("--pose_refine_log_every", type=int, default=25)
    p.add_argument(
        "--pose_refine_max_drot_deg",
        type=float,
        default=8.0,
        help="Max rotation drift (deg) for best-solution selection; limits overfitting to matches",
    )
    p.add_argument(
        "--pose_refine_max_dt_m",
        type=float,
        default=0.06,
        help="Max translation drift (m) for best-solution selection; limits overfitting to matches",
    )

    # Dense fusion
    p.add_argument("--dense_refine_iters", type=int, default=1)
    p.add_argument("--dense_refine_lr", type=float, default=0.0)
    p.add_argument(
        "--dense_conf_thr",
        type=float,
        default=12,
        help="Dense confidence threshold; lower keeps more points (can reduce large low-conf regions)",
    )
    p.add_argument(
        "--confidence_percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to normalize confidence visualizations",
    )
    p.add_argument("--voxel_size", type=float, default=0.0015)
    p.add_argument("--max_points", type=int, default=2_000_000)

    # Rerun 
    p.add_argument("--grpc_port", type=int, default=9876)
    p.add_argument("--no_wait", action="store_true")

    return p.parse_args()


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
        # A maps original pixel coordinates to MASt3R network pixel coordinates.
        A = np.array(
            [
                [self.scale_x, 0.0, -self.crop_left],
                [0.0, self.scale_y, -self.crop_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A @ K_orig.astype(np.float32)

    def network_to_original(self, uv_net: np.ndarray) -> np.ndarray:
        uv_orig = uv_net.astype(np.float32).copy()
        uv_orig[:, 0] = (uv_orig[:, 0] + self.crop_left) / self.scale_x
        uv_orig[:, 1] = (uv_orig[:, 1] + self.crop_top) / self.scale_y
        return uv_orig

    def network_grid_to_original(self):
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


def _log_step(message: str) -> None:
    print(f"[bold cyan]{message}[/bold cyan]")


def _compute_image_geometry(
    image: torch.Tensor,
    image_size: int,
    patch_size: int,
    square_ok: bool = False,
) -> ImageGeometry:
    H0, W0 = int(image.shape[1]), int(image.shape[2])
    if image_size == 224:
        raise ValueError("This pipeline expects image_size != 224 so that the MASt3R crop can be tracked exactly.")

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


def _fix_3x4_to_4x4(T: np.ndarray) -> np.ndarray:
    if T.shape == (4, 4):
        return T.astype(np.float32)
    if T.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = T.astype(np.float32)
        return out
    raise ValueError(f"Expected (4,4) or (3,4), got {T.shape}")


def _save_pointcloud_as_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None = None,
) -> None:
    assert points.shape[0] == colors.shape[0]
    if confidence is not None:
        assert confidence.shape[0] == points.shape[0]

    num_points = points.shape[0]
    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if confidence is not None:
        header.append("property float confidence")
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if confidence is None:
            data = np.concatenate([pts, cols.astype(np.float32)], axis=1)
            np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")
        else:
            conf = confidence.reshape(-1, 1).astype(np.float32)
            data = np.concatenate([pts, cols.astype(np.float32), conf], axis=1)
            np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d %.6f")


def _weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
):
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
    # init_from_known_poses expects every image index to be view1 in at least one edge
    if num_images >= 2:
        last_idx = num_images - 1
        if not any(_pair_first_idx(p) == last_idx for p in pairs):
            pairs.append((images[last_idx], images[last_idx - 1]))
    return pairs


def _pair_first_idx(pair) -> int:
    idx = pair[0]["idx"]
    return int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])


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
        "pred1": {
            "pts3d": pair_output["pred1"]["pts3d"],
            "conf": pair_output["pred1"]["conf"],
        },
        "pred2": {
            "pts3d_in_other_view": pair_output["pred2"]["pts3d_in_other_view"],
            "conf": pair_output["pred2"]["conf"],
        },
    }


@torch.no_grad()
def _match_pair_from_prediction(
    pred1,
    pred2,
    device,
    desc_conf_thr: float,
    max_matches_per_pair: int,
):
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
    # Clamp magnitude to pi so sin(theta)/theta and (1-cos)/theta^2 stay numerically stable.
    # Use eps 1e-8 so gradients exist and do not explode.
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
    B = torch.where(
        theta2 > eps,
        (1.0 - torch.cos(theta)) / theta2.clamp(min=eps),
        0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
    )
    return eye + A[..., None] * k + B[..., None] * kk


def _compose_refined_camera_poses(
    T_world_cam_init: torch.Tensor,
    rot_delta: torch.Tensor,
    trans_delta: torch.Tensor,
):
    # T_world_cam_init: [N,4,4], rot_delta/trans_delta: [N-1,3] for cameras 1..N-1.
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

    T_world_cam = torch.eye(4, dtype=T_world_cam_init.dtype, device=T_world_cam_init.device).unsqueeze(0).repeat(
        num_cams, 1, 1
    )
    T_world_cam[:, :3, :3] = R_init @ R_delta
    T_world_cam[:, :3, 3] = t_init + trans_full
    return T_world_cam, rot_full, trans_full


def _epipolar_error_px(
    T_world_cam: torch.Tensor,
    cam_i: torch.Tensor,
    cam_j: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    focal_px: float,
) -> torch.Tensor:
    R_world_cam = T_world_cam[:, :3, :3]
    t_world_cam = T_world_cam[:, :3, 3]

    R_i = R_world_cam[cam_i]  # [M,3,3]
    R_j = R_world_cam[cam_j]  # [M,3,3]
    t_i = t_world_cam[cam_i]  # [M,3]
    t_j = t_world_cam[cam_j]  # [M,3]

    R_ji = R_j.transpose(1, 2) @ R_i
    t_ji = (R_j.transpose(1, 2) @ (t_i - t_j).unsqueeze(-1)).squeeze(-1)
    E = _skew_symmetric(t_ji) @ R_ji

    Ex1 = (E @ x1.unsqueeze(-1)).squeeze(-1)
    Etx2 = (E.transpose(1, 2) @ x2.unsqueeze(-1)).squeeze(-1)
    x2tEx1 = (x2 * Ex1).sum(dim=-1)

    denom = (
        Ex1[:, :2].square().sum(dim=-1) + Etx2[:, :2].square().sum(dim=-1) + 1e-6
    )
    sampson = (x2tEx1.square() / denom).clamp(min=0.0)
    return torch.sqrt(sampson + 1e-6) * focal_px


def _prepare_pose_refinement_data(match_records, K: np.ndarray, device: torch.device):
    if len(match_records) == 0:
        return None

    cam_i_parts = []
    cam_j_parts = []
    pair_id_parts = []
    x1_parts = []
    x2_parts = []
    weight_parts = []
    pair_names = []
    pair_match_counts = []

    for pair_id, rec in enumerate(match_records):
        # x1/x2: [M,3] normalized homogeneous image coordinates for one camera pair.
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
    args,
    device: torch.device,
):
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

    if len(T_world_cam_init_np) < 2:
        print("[yellow]Skipping pose refinement: fewer than two cameras were provided.[/yellow]")
        return T_world_cam_init_np.astype(np.float32), pose_stats

    if len(match_records) == 0:
        print("[yellow]Skipping pose refinement: no valid MASt3R matches were produced.[/yellow]")
        return T_world_cam_init_np.astype(np.float32), pose_stats

    if args.pose_refine_iters <= 0:
        print("[yellow]Skipping pose refinement: --pose_refine_iters <= 0.[/yellow]")
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
            T_world_cam_init,
            problem["cam_i"],
            problem["cam_j"],
            problem["x1"],
            problem["x2"],
            focal_px=focal_px,
        )
        pose_stats["median_epipolar_px_before"] = float(err_before.median().item())
        pose_stats["mean_epipolar_px_before"] = float(err_before.mean().item())

    print(
        f"[green]Initial multi-view alignment error: median={pose_stats['median_epipolar_px_before']:.3f}px, "
        f"mean={pose_stats['mean_epipolar_px_before']:.3f}px over {problem['num_matches']:,} matches[/green]"
    )

    rot_delta = torch.zeros((len(T_world_cam_init_np) - 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros((len(T_world_cam_init_np) - 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rot_delta, trans_delta], lr=args.pose_refine_lr)

    best_median_px = float("inf")
    best_rot = rot_delta.detach().clone()
    best_trans = trans_delta.detach().clone()
    huber_beta = max(float(args.pixel_tol), 1e-4)
    sigma_rot_deg = max(float(args.pose_prior_sigma_deg), 1e-6)
    sigma_trans_m = max(float(args.pose_prior_sigma_m), 1e-6)
    max_rot_rad = np.pi
    max_trans_m = 0.2
    grad_clip_norm = 1.0

    def _cosine_lr(step: int, total: int, lr_max: float, lr_min: float) -> float:
        t = step / max(total - 1, 1)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t))

    _log_step("Step 4/8 - Refining camera poses with weighted multi-view epipolar alignment")
    for iteration in range(args.pose_refine_iters):
        lr = _cosine_lr(
            iteration,
            args.pose_refine_iters,
            float(args.pose_refine_lr),
            float(args.pose_refine_lr_min),
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        T_world_cam_refined, rot_full, trans_full = _compose_refined_camera_poses(
            T_world_cam_init, rot_delta, trans_delta
        )
        err_px = _epipolar_error_px(
            T_world_cam_refined,
            problem["cam_i"],
            problem["cam_j"],
            problem["x1"],
            problem["x2"],
            focal_px=focal_px,
        )
        robust_err = torch.where(
            err_px <= huber_beta,
            0.5 * err_px.square() / huber_beta,
            err_px - 0.5 * huber_beta,
        )

        pair_loss = torch.zeros((problem["num_pairs"],), dtype=torch.float32, device=device)
        pair_weight = torch.zeros((problem["num_pairs"],), dtype=torch.float32, device=device)
        pair_loss.index_add_(0, problem["pair_id"], robust_err * problem["weight"])
        pair_weight.index_add_(0, problem["pair_id"], problem["weight"])
        loss_epipolar = (pair_loss / pair_weight.clamp_min(1e-6)).mean()

        rot_norm_deg = torch.rad2deg(rot_delta.norm(dim=-1))
        trans_norm_m = trans_delta.norm(dim=-1)
        loss_prior = (
            (rot_norm_deg / sigma_rot_deg).square() + (trans_norm_m / sigma_trans_m).square()
        ).mean()
        loss = loss_epipolar + float(args.pose_prior_weight) * loss_prior

        median_px_iter = float(err_px.median().item()) if err_px.isfinite().any() else float("nan")
        max_drot_iter = float(rot_norm_deg.max().item()) if rot_norm_deg.isfinite().any() else float("inf")
        max_dt_iter = float(trans_norm_m.max().item()) if trans_norm_m.isfinite().any() else float("inf")
        within_cap = (
            max_drot_iter <= float(args.pose_refine_max_drot_deg)
            and max_dt_iter <= float(args.pose_refine_max_dt_m)
        )
        if (
            np.isfinite(median_px_iter)
            and median_px_iter < best_median_px
            and within_cap
        ):
            best_median_px = median_px_iter
            best_rot = rot_delta.detach().clone()
            best_trans = trans_delta.detach().clone()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([rot_delta, trans_delta], max_norm=grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            rot_norm = rot_delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            rot_delta.data.mul_(rot_norm.clamp(max=max_rot_rad) / rot_norm)
            trans_delta.data.clamp_(-max_trans_m, max_trans_m)
            rot_delta.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            trans_delta.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        should_log = (
            iteration == 0
            or (iteration + 1) % max(int(args.pose_refine_log_every), 1) == 0
            or (iteration + 1) == args.pose_refine_iters
        )
        if should_log:
            median_px = float(err_px.median().item()) if err_px.isfinite().any() else float("nan")
            max_dt = float(trans_norm_m.max().item()) if trans_norm_m.isfinite().any() else float("nan")
            max_drot = float(rot_norm_deg.max().item()) if rot_norm_deg.isfinite().any() else float("nan")
            loss_val = float(loss.item())
            grad_val = float(grad_norm.item()) if grad_norm.isfinite() else float("nan")
            print(
                f"[blue]Pose refinement {iteration + 1}/{args.pose_refine_iters}: "
                f"loss={loss_val:.4f}, "
                f"median_epipolar_px={median_px:.3f}, "
                f"grad_norm={grad_val:.4f}, "
                f"max_dt_m={max_dt:.5f}, max_drot_deg={max_drot:.3f}[/blue]"
            )

    with torch.no_grad():
        used_refinement = (best_rot.abs().sum().item() + best_trans.abs().sum().item()) > 1e-9
        if not used_refinement:
            print(
                "[yellow]No refinement within drift caps (--pose_refine_max_drot_deg/--pose_refine_max_dt_m); "
                "using initial poses. Consider increasing caps if poses are known to be rough.[/yellow]"
            )
        T_world_cam_best, rot_full_best, trans_full_best = _compose_refined_camera_poses(
            T_world_cam_init, best_rot, best_trans
        )
        err_after = _epipolar_error_px(
            T_world_cam_best,
            problem["cam_i"],
            problem["cam_j"],
            problem["x1"],
            problem["x2"],
            focal_px=focal_px,
        )

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

    print(
        f"[green]Refined multi-view alignment error: median={pose_stats['median_epipolar_px_after']:.3f}px, "
        f"mean={pose_stats['mean_epipolar_px_after']:.3f}px[/green]"
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


def _colorize_scalar_field(values: np.ndarray, percentile: float = 99.0, log_compress: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vis = np.log1p(np.clip(arr, a_min=0.0, a_max=None)) if log_compress else arr.copy()
    finite = np.isfinite(vis)

    if not np.any(finite):
        return np.zeros(arr.shape + (3,), dtype=np.uint8)

    vmax = float(np.percentile(vis[finite], percentile))
    if vmax <= 1e-12:
        vmax = float(vis[finite].max())
    if vmax <= 1e-12:
        return np.zeros(arr.shape + (3,), dtype=np.uint8)

    norm = np.clip(vis / vmax, 0.0, 1.0)
    values_u8 = (norm * 255.0).astype(np.uint8)
    if values_u8.ndim == 1:
        colored = cv2.applyColorMap(values_u8[:, None], cv2.COLORMAP_TURBO)[:, 0, :]
        colored = colored[:, ::-1]
        colored[~finite] = 0
        return colored
    else:
        colored = cv2.applyColorMap(values_u8, cv2.COLORMAP_TURBO)
        colored[~finite] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _fuse_dense_scene(
    scene,
    orig_images,
    geometries,
    voxel_size: float,
    max_points: int,
    confidence_percentile: float,
):
    with torch.no_grad():
        pts3d_list = [pts.detach().cpu().numpy() for pts in scene.get_pts3d()]
        conf_list = [conf.detach().cpu().numpy() for conf in scene.im_conf]
        mask_list = [mask.detach().cpu().numpy() for mask in scene.get_masks()]

    all_pts = []
    all_cols = []
    all_conf = []
    conf_maps_original = []
    conf_visualizations = []
    per_image_counts = []
    fallback_view_indices = []

    for image_idx, (pts3d, conf_map, mask, geom, image) in enumerate(
        zip(pts3d_list, conf_list, mask_list, geometries, orig_images)
    ):
        # pts3d: [H_net,W_net,3], conf_map/mask: [H_net,W_net].
        conf_orig = geom.warp_network_map_to_original(conf_map)
        conf_maps_original.append(conf_orig)
        conf_visualizations.append(
            _colorize_scalar_field(conf_orig, percentile=confidence_percentile, log_compress=True)
        )

        uu_orig, vv_orig = geom.network_grid_to_original()
        rgb = image.permute(1, 2, 0).cpu().numpy()
        u_idx = np.clip(np.round(uu_orig).astype(np.int32), 0, geom.original_width - 1)
        v_idx = np.clip(np.round(vv_orig).astype(np.int32), 0, geom.original_height - 1)
        cols_full = rgb[v_idx, u_idx].astype(np.uint8)

        valid = mask & np.isfinite(pts3d).all(axis=-1)
        if not np.any(valid):
            valid = (conf_map > 0) & np.isfinite(pts3d).all(axis=-1)
            fallback_view_indices.append(image_idx)

        pts = pts3d[valid].astype(np.float32)
        cols = cols_full[valid].astype(np.uint8)
        conf = conf_map[valid].astype(np.float32)

        per_image_counts.append(int(len(pts)))
        all_pts.append(pts)
        all_cols.append(cols)
        all_conf.append(conf)
        print(f"[green]Dense fusion image {image_idx}: kept {len(pts):,} points[/green]")

    total_raw_points = int(sum(per_image_counts))
    if total_raw_points == 0:
        raise RuntimeError("No valid dense points were produced by the MASt3R fusion stage.")

    final_pts = np.concatenate(all_pts, axis=0)
    final_cols = np.concatenate(all_cols, axis=0)
    final_conf = np.concatenate(all_conf, axis=0)

    if voxel_size > 0:
        final_pts, final_cols, final_conf = _weighted_voxel_downsample(
            final_pts, final_cols, final_conf, voxel_size
        )

    if len(final_pts) > max_points:
        keep = np.argsort(final_conf)[-max_points:][::-1]
        final_pts = final_pts[keep]
        final_cols = final_cols[keep]
        final_conf = final_conf[keep]

    point_conf_colors = _colorize_scalar_field(
        final_conf.astype(np.float32),
        percentile=confidence_percentile,
        log_compress=True,
    )

    return {
        "points": final_pts,
        "colors": final_cols,
        "confidence": final_conf,
        "confidence_colors": point_conf_colors,
        "confidence_maps_original": conf_maps_original,
        "confidence_visualizations": conf_visualizations,
        "per_image_counts": per_image_counts,
        "num_points_before_downsample": total_raw_points,
        "fallback_view_indices": fallback_view_indices,
    }


def main():
    args = _parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", f"[green]{device}[/green]")
    if device.type != "cuda":
        print("[yellow]CUDA was not detected; the pipeline will still run but it will be much slower.[/yellow]")

    _log_step("Step 1/8 - Loading RGB images, poses, and calibration")
    image_files = sorted(args.rgb_dir.glob("image_*.png"))
    pose_files = sorted(args.pose_dir.glob("pose_*.txt")) if args.pose_dir.exists() else []
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.rgb_dir}")
    if len(image_files) < 2:
        raise RuntimeError("This reconstruction pipeline needs at least two images.")

    # GT poses are optional - if missing, we let MASt3R estimate them via MST init.
    have_gt_poses = len(pose_files) == len(image_files)
    if len(pose_files) > 0 and not have_gt_poses:
        print(
            f"[yellow]Found {len(image_files)} images but {len(pose_files)} pose files; "
            "ignoring pose files and running in RGB-only mode.[/yellow]"
        )
    elif len(pose_files) == 0:
        print(
            "[yellow]No pose files found; MASt3R will estimate camera poses from scratch.[/yellow]"
        )

    orig_images = [io.read_image(str(path)) for path in image_files]  # [3,H,W], uint8, CPU

    # GT intrinsics are optional - if missing, MASt3R will estimate focals/principal points.
    intrinsics_path = args.camera_params_dir / "intrinsics.npy"
    cam2ee_path = args.camera_params_dir / "camera2ee.npy"
    have_gt_intrinsics = intrinsics_path.exists()
    if have_gt_intrinsics:
        K_orig = np.load(intrinsics_path).astype(np.float32)
    else:
        K_orig = None
        print(
            f"[yellow]{intrinsics_path} not found; MASt3R will estimate intrinsics.[/yellow]"
        )

    if have_gt_poses:
        poses6 = []
        for pose_file in pose_files:
            pose_values = [float(x) for x in pose_file.read_text().strip().split()]
            poses6.append(pose_values)
        poses6 = torch.tensor(poses6, dtype=torch.float32, device=device)  # [N,6]

        T_world_ee = xyzeuler_to_hmat(
            poses6,
            convention="ROLLPITCHYAW",
            translation_scale=1.0,
        )  # [N,4,4], assumed to be in meters.

        if cam2ee_path.exists():
            T_ee_cam = _fix_3x4_to_4x4(np.load(cam2ee_path))
            print(f"Loaded camera-to-EE transform from [green]{cam2ee_path}[/green]")
        else:
            T_ee_cam = np.eye(4, dtype=np.float32)
            print("[yellow]camera2ee.npy not found, assuming poses are already camera poses.[/yellow]")

        T_ee_cam_t = (
            torch.tensor(T_ee_cam, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(poses6), 1, 1)
        )  # [N,4,4]
        T_world_cam_init = torch.matmul(T_world_ee, T_ee_cam_t).detach().cpu().numpy().astype(np.float32)
    else:
        T_world_cam_init = None

    print(
        f"Loaded {len(image_files)} images"
        + (f" and {len(pose_files)} poses" if have_gt_poses else " (RGB-only)")
    )

    _log_step("Step 2/8 - Initializing Rerun and loading the MASt3R model")
    rr.init("SpectraBreast_MASt3R_MetricDenseFusion")
    rr.serve_grpc(grpc_port=args.grpc_port)

    repo_root = Path(__file__).resolve().parent
    mast3r_root = repo_root / "mast3r"
    if str(mast3r_root) not in sys.path:
        sys.path.append(str(mast3r_root))

    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.inference import inference
    from dust3r.utils.device import collate_with_cat
    from dust3r.utils.image import load_images
    from mast3r.model import AsymmetricMASt3R

    print(f"Loading MASt3R model: {args.model_name}")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)
    model.eval()
    patch_size = int(getattr(model, "patch_size", 16))
    print(f"Using MASt3R patch size: {patch_size}")

    _log_step("Step 3/8 - Preparing MASt3R image geometry and running pairwise inference")
    filelist = [str(path) for path in image_files]
    imgs_mast3r = load_images(filelist, size=args.image_size, verbose=True, patch_size=patch_size)
    geometries = [
        _compute_image_geometry(image, image_size=args.image_size, patch_size=patch_size, square_ok=False)
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

    pairs = _make_temporal_pairs(imgs_mast3r, args.neighbor_window)
    if len(pairs) == 0:
        raise RuntimeError("No temporal pairs were generated. Increase --neighbor_window.")
    print(f"Built {len(pairs)} temporal image pairs")

    compact_pair_results = []
    match_records = []

    for pair_idx, (img_i, img_j) in enumerate(pairs):
        image_i = int(img_i["idx"])
        image_j = int(img_j["idx"])
        print(f"[cyan]Running MASt3R on pair {pair_idx + 1}/{len(pairs)} -> ({image_i}, {image_j})[/cyan]")

        pair_output = inference([(img_i, img_j)], model, device, batch_size=1, verbose=False)

        uv_i_net, uv_j_net, match_conf = _match_pair_from_prediction(
            pair_output["pred1"],
            pair_output["pred2"],
            device=device,
            desc_conf_thr=args.desc_conf_thr,
            max_matches_per_pair=args.max_matches_per_pair,
        )
        if uv_i_net is None or len(uv_i_net) == 0:
            print(f"[yellow]Pair ({image_i}, {image_j}): no valid reciprocal descriptor matches[/yellow]")
        else:
            uv_i_orig = geometries[image_i].network_to_original(uv_i_net)
            uv_j_orig = geometries[image_j].network_to_original(uv_j_net)
            match_records.append(
                {
                    "i": image_i,
                    "j": image_j,
                    "uv_i": uv_i_orig,
                    "uv_j": uv_j_orig,
                    "conf": match_conf,
                }
            )
            print(
                f"[green]Pair ({image_i}, {image_j}): extracted {len(match_conf):,} metric calibration matches[/green]"
            )

        compact_pair_results.append(_compact_pair_output(pair_output))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    multiple_shapes = not _pairs_share_same_network_shape(pairs)
    compact_output = collate_with_cat(compact_pair_results, lists=multiple_shapes)

    if have_gt_poses and have_gt_intrinsics:
        refined_poses, pose_stats = _refine_camera_poses_with_epipolar(
            match_records=match_records,
            T_world_cam_init_np=T_world_cam_init,
            K_orig=K_orig,
            args=args,
            device=device,
        )
    else:
        # Epipolar pose refinement requires both GT poses (initialization) and
        # GT intrinsics (to build the essential matrix). Skip it otherwise.
        refined_poses = T_world_cam_init  # may be None
        pose_stats = {
            "pairs_used": len(match_records),
            "matches_used": int(sum(len(rec["conf"]) for rec in match_records)),
            "used_alignment": False,
            "note": "Skipped epipolar refinement (RGB-only mode or missing intrinsics)",
        }

    _log_step("Step 5/8 - Running dense MASt3R fusion")
    _patch_modular_optimizer()
    scene = global_aligner(
        compact_output,
        device=device,
        mode=GlobalAlignerMode.ModularPointCloudOptimizer,
        verbose=True,
        min_conf_thr=args.dense_conf_thr,
        # fx_and_fy requires both focals to be preset exactly; only enable with GT K.
        fx_and_fy=bool(have_gt_intrinsics),
    )

    if have_gt_poses:
        refined_poses_t = torch.tensor(refined_poses, dtype=torch.float32, device=device)
        scene.preset_pose(refined_poses_t)
    if have_gt_intrinsics:
        _preset_exact_intrinsics(scene.cpu(), K_network)

    dense_init = "known_poses" if have_gt_poses else "mst"
    dense_loss = scene.compute_global_alignment(
        init=dense_init,
        niter=args.dense_refine_iters,
        schedule="cosine",
        lr=args.dense_refine_lr,
    )
    print(f"[green]Dense MASt3R optimization finished with final loss {float(dense_loss):.6f}[/green]")

    # After alignment, always export the cameras that MASt3R actually used for
    # fusion. In RGB-only mode these are MASt3R's estimated cam-to-world poses.
    with torch.no_grad():
        refined_poses = scene.get_im_poses().detach().cpu().numpy().astype(np.float32)
        scene_intrinsics_network = scene.get_intrinsics().detach().cpu().numpy().astype(np.float32)
    if K_network is None:
        K_network = [K for K in scene_intrinsics_network]

    _log_step("Step 6/8 - Cleaning the dense reconstruction and building a single fused point cloud")
    scene = scene.clean_pointcloud()
    dense_conf_thr_raw = float(scene.conf_trf(torch.tensor(args.dense_conf_thr)))
    scene.min_conf_thr = dense_conf_thr_raw
    print(f"Using dense confidence mask threshold: user={args.dense_conf_thr}, raw={dense_conf_thr_raw:.6f}")
    print(
        "[dim]Large low-conf regions in some views can be due to pose errors or textureless areas; "
        "lower --dense_conf_thr to keep more points from those regions.[/dim]"
    )

    fusion_result = _fuse_dense_scene(
        scene=scene,
        orig_images=orig_images,
        geometries=geometries,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        confidence_percentile=args.confidence_percentile,
    )
    final_pts = fusion_result["points"]
    final_cols = fusion_result["colors"]
    final_conf = fusion_result["confidence"]
    point_conf_colors = fusion_result["confidence_colors"]

    print(f"Raw dense cloud before voxelization: {fusion_result['num_points_before_downsample']:,} points")
    print(f"Final fused cloud: {len(final_pts):,} points")
    if fusion_result["fallback_view_indices"]:
        print(
            "[yellow]Dense fusion fallback (conf mask empty) for view(s): "
            f"{fusion_result['fallback_view_indices']}. Consider lowering --dense_conf_thr.[/yellow]"
        )

    _log_step("Step 7/8 - Logging cameras, point cloud, RGB, and confidence maps to Rerun")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = args.out_dir / "confidence_maps"
    conf_dir.mkdir(parents=True, exist_ok=True)

    for image_idx, (image, T_world_cam, geom, conf_map, conf_vis) in enumerate(
        zip(
            orig_images,
            refined_poses,
            geometries,
            fusion_result["confidence_maps_original"],
            fusion_result["confidence_visualizations"],
        )
    ):
        rgb = image.permute(1, 2, 0).cpu().numpy()
        rr.log(
            f"/cameras/{image_idx}",
            rr.Transform3D(
                translation=T_world_cam[:3, 3],
                mat3x3=T_world_cam[:3, :3],
            ),
        )
        if have_gt_intrinsics:
            pinhole_K = K_orig
            pinhole_res = [geom.original_width, geom.original_height]
        else:
            # In RGB-only mode MASt3R intrinsics live in the network crop geometry,
            # so log against the network-resolution image.
            pinhole_K = np.asarray(K_network[image_idx], dtype=np.float32)
            pinhole_res = [geom.network_width, geom.network_height]
        rr.log(
            f"/cameras/{image_idx}/image",
            rr.Pinhole(
                image_from_camera=pinhole_K,
                resolution=pinhole_res,
                image_plane_distance=0.01,
            ),
        )
        rr.log(f"/cameras/{image_idx}/image/rgb", rr.Image(rgb))
        rr.log(f"/cameras/{image_idx}/image/confidence", rr.Image(conf_vis))

        np.save(conf_dir / f"confidence_{image_idx:03d}.npy", conf_map.astype(np.float32))
        cv2.imwrite(
            str(conf_dir / f"confidence_{image_idx:03d}.png"),
            cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR),
        )

    rr.log("/points", rr.Points3D(final_pts, colors=final_cols))
    rr.log("/points_confidence", rr.Points3D(final_pts, colors=point_conf_colors))
    _send_rerun_blueprint()

    _log_step("Step 8/8 - Saving the metric reconstruction and metadata")
    ply_path = args.out_dir / "mast3r_metric_aligned_dense.ply"
    poses_path = args.out_dir / "refined_camera_poses.npy"
    network_intrinsics_path = args.out_dir / "intrinsics_network.npy"
    cloud_npz_path = args.out_dir / "mast3r_metric_aligned_dense.npz"
    params_path = args.out_dir / "mast3r_metric_aligned_dense_params.json"

    _save_pointcloud_as_ply(ply_path, final_pts, final_cols, confidence=final_conf)
    np.save(poses_path, refined_poses.astype(np.float32))
    np.save(network_intrinsics_path, np.stack(K_network, axis=0).astype(np.float32))
    np.savez(
        cloud_npz_path,
        points=final_pts.astype(np.float32),
        colors=final_cols.astype(np.uint8),
        confidence=final_conf.astype(np.float32),
    )

    params = {
        "rgb_dir": str(args.rgb_dir),
        "pose_dir": str(args.pose_dir),
        "camera_params_dir": str(args.camera_params_dir),
        "out_dir": str(args.out_dir),
        "model_name": args.model_name,
        "image_size": args.image_size,
        "neighbor_window": args.neighbor_window,
        "desc_conf_thr": args.desc_conf_thr,
        "max_matches_per_pair": args.max_matches_per_pair,
        "pixel_tol": args.pixel_tol,
        "pose_refine_iters": args.pose_refine_iters,
        "pose_refine_lr": args.pose_refine_lr,
        "pose_refine_lr_min": args.pose_refine_lr_min,
        "pose_refine_max_drot_deg": args.pose_refine_max_drot_deg,
        "pose_refine_max_dt_m": args.pose_refine_max_dt_m,
        "pose_prior_sigma_deg": args.pose_prior_sigma_deg,
        "pose_prior_sigma_m": args.pose_prior_sigma_m,
        "pose_prior_weight": args.pose_prior_weight,
        "dense_refine_iters": args.dense_refine_iters,
        "dense_refine_lr": args.dense_refine_lr,
        "dense_conf_thr_user": args.dense_conf_thr,
        "dense_conf_thr_raw": dense_conf_thr_raw,
        "voxel_size": args.voxel_size,
        "max_points": args.max_points,
        "confidence_percentile": args.confidence_percentile,
        "num_images": len(image_files),
        "num_pairs": len(pairs),
        "num_alignment_pairs": len(match_records),
        "num_points_before_downsample": fusion_result["num_points_before_downsample"],
        "num_points_output": int(len(final_pts)),
        "pointcloud_units": "meters",
        "output_ply": str(ply_path),
        "output_npz": str(cloud_npz_path),
        "output_refined_camera_poses": str(poses_path),
        "output_network_intrinsics": str(network_intrinsics_path),
        "output_confidence_dir": str(conf_dir),
        "per_image_dense_point_counts": fusion_result["per_image_counts"],
        "fallback_view_indices": fusion_result["fallback_view_indices"],
        "pose_refinement": pose_stats,
        "have_gt_poses": bool(have_gt_poses),
        "have_gt_intrinsics": bool(have_gt_intrinsics),
        "dense_init": dense_init,
        "camera2ee_used": str(cam2ee_path) if (have_gt_poses and cam2ee_path.exists()) else None,
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"[green]Saved metric point cloud to {ply_path}[/green]")
    print(f"[green]Saved refined camera poses to {poses_path}[/green]")
    print(f"[green]Saved confidence maps to {conf_dir}[/green]")
    print(f"[green]Saved metadata to {params_path}[/green]")

    if not args.no_wait:
        input("Data has been logged to Rerun. Open the viewer now and then press enter.")


if __name__ == "__main__":
    main()