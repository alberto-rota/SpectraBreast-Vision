import argparse
import json
import sys
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
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/",
                    contents=["/points", "/cameras_gt/**", "/cameras_reconstruction/**"],
                ),
                rrb.Spatial3DView(
                    name="Confidence",
                    origin="/",
                    contents=["/points_confidence"],
                ),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="Camera RGB",
                    origin="/2d/rgb",
                ),
                rrb.Spatial2DView(
                    name="Camera Confidence",
                    origin="/2d/confidence",
                ),
            ),
        ),
    )
    rr.send_blueprint(blueprint, make_active=True)


def _parse_args():
    p = argparse.ArgumentParser(description="VGGT Point Cloud Extraction")
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction_vggt"))

    # VGGT
    p.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence percentile to filter out (e.g. 50 filters out bottom 50%)",
    )

    # Options
    p.add_argument(
        "--cloud_source",
        type=str,
        choices=["point_map", "depth_map"],
        default="depth_map",
        help="Source of the 3D points: 'point_map' branch directly, or 'depth_map' unprojection",
    )
    p.add_argument(
        "--camera_source",
        type=str,
        choices=["predicted", "gt"],
        default="predicted",
        help=(
            "'predicted': keep VGGT reconstruction in its native predicted frame. "
            "'gt': reconstruct with predicted cameras, then align the full reconstruction "
            "to the GT robot/world frame."
        ),
    )
    p.add_argument(
        "--alignment_mode",
        type=str,
        choices=["sim3", "se3"],
        default="sim3",
        help="Alignment used when --camera_source gt. sim3 is usually the right choice.",
    )

    p.add_argument("--mask_black_bg", action="store_true", help="Mask out black background pixels")
    p.add_argument("--mask_white_bg", action="store_true", help="Mask out white background pixels")

    # Rerun
    p.add_argument("--grpc_port", type=int, default=9876)
    p.add_argument("--no_wait", action="store_true")

    return p.parse_args()


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


def _log_step(message: str) -> None:
    print(f"[bold cyan]{message}[/bold cyan]")


def _camera_from_world_extrinsics_to_world_from_camera(T_cam_world_3x4: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV extrinsics [R|t] mapping world->camera into 4x4 camera->world poses.
    Input:  [S, 3, 4]
    Output: [S, 4, 4]
    """
    T_world_cam = []
    for i in range(len(T_cam_world_3x4)):
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :4] = T_cam_world_3x4[i].astype(np.float32)
        T_wc = np.linalg.inv(T_cw).astype(np.float32)
        T_world_cam.append(T_wc)
    return np.stack(T_world_cam, axis=0)


def _fit_umeyama_alignment(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    estimate_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Find similarity transform such that:
        x_dst ~= s * R * x_src + t
    using Umeyama alignment.
    """
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src,dst shape [N,3], got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate a stable similarity transform")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1.0

    R = U @ np.diag(sign) @ Vt

    if estimate_scale:
        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        if var_src < 1e-12:
            raise ValueError("Degenerate source configuration while estimating similarity transform")
        scale = float(np.sum(D * sign) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R.astype(np.float32), t.astype(np.float32)


def _apply_similarity_transform(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Applies x' = s * R * x + t
    points can be [..., 3]
    """
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape
    pts_flat = pts.reshape(-1, 3).astype(np.float64)
    out = scale * (pts_flat @ R.astype(np.float64).T) + t.astype(np.float64)[None, :]
    return out.astype(np.float32).reshape(orig_shape)


def _apply_similarity_to_camera_poses(
    T_world_cam: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> np.ndarray:
    """
    Transform camera->world poses under x' = s * R_align * x + t_align.
    The camera center is transformed with the full Sim(3).
    The camera orientation is rotated by R_align.
    """
    T_world_cam = np.asarray(T_world_cam, dtype=np.float32)
    out = np.repeat(np.eye(4, dtype=np.float32)[None], T_world_cam.shape[0], axis=0)

    for i in range(T_world_cam.shape[0]):
        R_wc = T_world_cam[i, :3, :3]
        t_wc = T_world_cam[i, :3, 3]

        out[i, :3, :3] = R_align @ R_wc
        out[i, :3, 3] = scale * (R_align @ t_wc) + t_align

    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main():
    args = _parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", f"[green]{device}[/green]")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    _log_step("Step 1/7 - Loading RGB images, poses, and calibration")
    image_files = sorted(args.rgb_dir.glob("image_*.png"))
    pose_files = sorted(args.pose_dir.glob("pose_*.txt"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.rgb_dir}")
    if len(image_files) != len(pose_files):
        raise RuntimeError(f"Found {len(image_files)} images but {len(pose_files)} pose files")

    orig_images = [io.read_image(str(path)) for path in image_files]  # [3,H,W], uint8

    poses6 = []
    for pose_file in pose_files:
        pose_values = [float(x) for x in pose_file.read_text().strip().split()]
        poses6.append(pose_values)
    poses6 = torch.tensor(poses6, dtype=torch.float32, device=device)

    T_world_ee = xyzeuler_to_hmat(
        poses6,
        convention="ROLLPITCHYAW",
        translation_scale=1.0,
    )

    K_orig = np.load(args.camera_params_dir / "intrinsics.npy").astype(np.float32)
    cam2ee_path = args.camera_params_dir / "camera2ee.npy"
    if cam2ee_path.exists():
        T_ee_cam = _fix_3x4_to_4x4(np.load(cam2ee_path))
        print(f"Loaded camera-to-EE transform from [green]{cam2ee_path}[/green]")
    else:
        T_ee_cam = np.eye(4, dtype=np.float32)
        print("[yellow]camera2ee.npy not found, assuming poses are already camera poses.[/yellow]")

    T_ee_cam_t = torch.tensor(T_ee_cam, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(poses6), 1, 1)
    T_world_cam_gt = torch.matmul(T_world_ee, T_ee_cam_t).detach().cpu().numpy().astype(np.float32)

    print(f"Loaded {len(image_files)} images and {len(pose_files)} poses")

    _log_step("Step 2/7 - Initializing Rerun and loading the VGGT model")
    rr.init("SpectraBreast_VGGT_Extraction")
    rr.serve_grpc(grpc_port=args.grpc_port)

    repo_root = Path(__file__).resolve().parent
    vggt_root = repo_root / "vggt"
    if str(vggt_root) not in sys.path:
        sys.path.append(str(vggt_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    print(f"Loading VGGT model: {args.model_name}")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()

    _log_step("Step 3/7 - Preparing VGGT image geometry")
    images_vggt = load_and_preprocess_images([str(p) for p in image_files], mode="crop").to(device)

    geometries = []
    target_size = args.image_size
    for img in orig_images:
        H, W = int(img.shape[1]), int(img.shape[2])
        new_width = target_size
        new_height = round(H * (new_width / W) / 14) * 14

        scale_x = new_width / W
        scale_y = new_height / H

        start_y = (new_height - target_size) // 2 if new_height > target_size else 0
        crop_h = target_size if new_height > target_size else new_height

        K_net = np.eye(3, dtype=np.float32)
        K_net[0, 0] = K_orig[0, 0] * scale_x
        K_net[1, 1] = K_orig[1, 1] * scale_y
        K_net[0, 2] = K_orig[0, 2] * scale_x
        K_net[1, 2] = K_orig[1, 2] * scale_y - start_y

        geometries.append(
            {
                "original_width": W,
                "original_height": H,
                "new_width": new_width,
                "new_height": new_height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "start_y": start_y,
                "crop_h": crop_h,
                "K_net": K_net,
            }
        )

    _log_step(
        f"Step 4/7 - Running VGGT inference "
        f"(Cloud source: {args.cloud_source}, Output frame: {args.camera_source})"
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_vggt_batch = images_vggt.unsqueeze(0)  # [1, S, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(images_vggt_batch)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic_pred, intrinsic_pred = pose_encoding_to_extri_intri(pose_enc, images_vggt_batch.shape[-2:])

            # IMPORTANT:
            # Always reconstruct using VGGT's predicted camera geometry.
            if args.cloud_source == "depth_map":
                depth_map, map_conf = model.depth_head(aggregated_tokens_list, images_vggt_batch, ps_idx)
                point_map = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    extrinsic_pred.squeeze(0),
                    intrinsic_pred.squeeze(0),
                )  # numpy [S, H, W, 3]
            else:
                point_map_pred, map_conf = model.point_head(aggregated_tokens_list, images_vggt_batch, ps_idx)
                point_map = point_map_pred.squeeze(0)[..., :3].float().cpu().numpy()  # [S, H, W, 3]

            map_conf = map_conf.squeeze(0).float().cpu().numpy()  # [S, H, W]
            intrinsic_pred_np = intrinsic_pred.squeeze(0).float().cpu().numpy()  # [S, 3, 3]
            extrinsic_pred_np = extrinsic_pred.squeeze(0).float().cpu().numpy()  # [S, 3, 4]

    T_world_cam_pred = _camera_from_world_extrinsics_to_world_from_camera(extrinsic_pred_np)
    T_world_cam_reconstruction = T_world_cam_pred.copy()
    point_map_in_output_frame = point_map.copy()

    alignment_info = None
    similarity_path = None

    _log_step("Step 5/7 - Aligning reconstruction frame if requested")
    if args.camera_source == "gt":
        pred_centers = T_world_cam_pred[:, :3, 3]
        gt_centers = T_world_cam_gt[:, :3, 3]

        estimate_scale = args.alignment_mode == "sim3"
        scale_align, R_align, t_align = _fit_umeyama_alignment(
            pred_centers,
            gt_centers,
            estimate_scale=estimate_scale,
        )

        pred_centers_aligned = _apply_similarity_transform(pred_centers, scale_align, R_align, t_align)
        center_rmse_before = _rmse(pred_centers, gt_centers)
        center_rmse_after = _rmse(pred_centers_aligned, gt_centers)

        print(
            f"[green]Alignment estimated ({args.alignment_mode})[/green]\n"
            f"  scale: {scale_align:.6f}\n"
            f"  camera-center RMSE before: {center_rmse_before:.6f} m\n"
            f"  camera-center RMSE after : {center_rmse_after:.6f} m"
        )

        point_map_in_output_frame = _apply_similarity_transform(point_map, scale_align, R_align, t_align)
        T_world_cam_reconstruction = _apply_similarity_to_camera_poses(
            T_world_cam_pred, scale_align, R_align, t_align
        )

        alignment_info = {
            "mode": args.alignment_mode,
            "scale": float(scale_align),
            "rotation_matrix": R_align.tolist(),
            "translation": t_align.tolist(),
            "camera_center_rmse_before_m": center_rmse_before,
            "camera_center_rmse_after_m": center_rmse_after,
        }
    else:
        print("[yellow]Keeping reconstruction in VGGT predicted frame[/yellow]")

    _log_step("Step 6/7 - Consolidating valid 3D points")
    all_pts = []
    all_cols = []
    all_conf = []
    conf_visualizations = []

    images_np = images_vggt.permute(0, 2, 3, 1).cpu().numpy()  # [S, H_net, W_net, 3]

    for img_idx in range(len(image_files)):
        P_world = point_map_in_output_frame[img_idx]  # [H, W, 3]
        conf = map_conf[img_idx]  # [H, W]
        img_rgb = images_np[img_idx]  # [H, W, 3] values 0..1
        cols = (img_rgb * 255).astype(np.uint8)

        if args.conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, args.conf_thres)

        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

        if args.mask_black_bg:
            black_bg_mask = cols.sum(axis=2) >= 16
            conf_mask = conf_mask & black_bg_mask

        if args.mask_white_bg:
            white_bg_mask = ~(
                (cols[:, :, 0] > 240)
                & (cols[:, :, 1] > 240)
                & (cols[:, :, 2] > 240)
            )
            conf_mask = conf_mask & white_bg_mask

        valid_points = np.isfinite(P_world).all(axis=-1) & (np.linalg.norm(P_world, axis=-1) > 1e-6)
        valid = conf_mask & valid_points

        conf_vis = _colorize_scalar_field(conf, percentile=99.0, log_compress=True)
        conf_visualizations.append(conf_vis)

        if not np.any(valid):
            continue

        P_valid = P_world[valid]
        conf_valid = conf[valid]
        cols_valid = cols[valid]

        all_pts.append(P_valid.astype(np.float32))
        all_cols.append(cols_valid.astype(np.uint8))
        all_conf.append(conf_valid.astype(np.float32))

        print(f"[green]Point extraction image {img_idx}: kept {len(P_valid):,} points[/green]")

    if not all_pts:
        raise RuntimeError("No valid points were extracted. Check your confidence threshold.")

    final_pts = np.concatenate(all_pts, axis=0)
    final_cols = np.concatenate(all_cols, axis=0)
    final_conf = np.concatenate(all_conf, axis=0)

    print(f"Total extracted points: {len(final_pts):,}")

    point_conf_colors = _colorize_scalar_field(
        final_conf.astype(np.float32),
        percentile=99.0,
        log_compress=True,
    )

    _log_step("Step 7/7 - Logging and saving the reconstruction")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = args.out_dir / "confidence_maps"
    conf_dir.mkdir(parents=True, exist_ok=True)

    rr.log("/points", rr.Points3D(final_pts, colors=final_cols), static=True)
    rr.log("/points_confidence", rr.Points3D(final_pts, colors=point_conf_colors), static=True)

    for image_idx, (image_rgb, T_gt, T_rec, K_rec, geom, conf_vis) in enumerate(
        zip(
            images_np,
            T_world_cam_gt,
            T_world_cam_reconstruction,
            intrinsic_pred_np,
            geometries,
            conf_visualizations,
        )
    ):
        # GT cameras (Green)
        rr.log(
            f"/cameras_gt/{image_idx}",
            rr.Transform3D(
                translation=T_gt[:3, 3],
                mat3x3=T_gt[:3, :3],
            ),
            static=True,
        )
        rr.log(
            f"/cameras_gt/{image_idx}/image",
            rr.Pinhole(
                image_from_camera=geom["K_net"],
                resolution=[image_rgb.shape[1], image_rgb.shape[0]],
                image_plane_distance=0.005,
                color=[0, 255, 0],
            ),
            static=True,
        )
        # rr.log(
        #     f"/cameras_gt/{image_idx}/image",
        #     rr.components.Color([0, 255, 0]),
        #     static=True,
        # )

        # Reconstruction cameras: (Red)
        # - predicted frame if camera_source=predicted
        # - aligned predicted cameras if camera_source=gt
        rr.log(
            f"/cameras_reconstruction/{image_idx}",
            rr.Transform3D(
                translation=T_rec[:3, 3],
                mat3x3=T_rec[:3, :3],
            ),
            static=True,
        )
        rr.log(
            f"/cameras_reconstruction/{image_idx}/image",
            rr.Pinhole(
                image_from_camera=geom["K_net"],
                resolution=[image_rgb.shape[1], image_rgb.shape[0]],
                image_plane_distance=0.005,
                color=[255, 0, 0],
            ),
            static=True,
        )
        # rr.log(
        #     f"/cameras_reconstruction/{image_idx}/image",
        #     rr.components.Color([255, 0, 0]),
        #     static=True,
        # )

        rr.set_time("frame_idx", sequence=image_idx)
        img_tensor = (image_rgb * 255).astype(np.uint8)
        
        # Log the image to the 3D cameras dynamically so they light up for the current frame
        # rr.log(f"/cameras_gt/{image_idx}/image", rr.Image(img_tensor))
        # rr.log(f"/cameras_reconstruction/{image_idx}/image", rr.Image(img_tensor))

        # Log to the 2D timeline explicitly
        rr.log("/2d/rgb", rr.Image(img_tensor))
        rr.log("/2d/confidence", rr.Image(conf_vis))

        cv2.imwrite(
            str(conf_dir / f"confidence_{image_idx:03d}.png"),
            cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR),
        )

    _send_rerun_blueprint()

    ply_path = args.out_dir / "vggt_extracted_cloud.ply"
    cloud_npz_path = args.out_dir / "vggt_extracted_cloud.npz"
    params_path = args.out_dir / "vggt_extracted_cloud_params.json"

    # Canonical output camera poses:
    # - predicted mode -> save predicted reconstruction cameras
    # - gt mode -> save GT camera poses, because the final cloud is intended to be in the GT/world frame
    poses_path = args.out_dir / "camera_poses_output_frame.npy"
    if args.camera_source == "gt":
        np.save(poses_path, T_world_cam_gt.astype(np.float32))
    else:
        np.save(poses_path, T_world_cam_pred.astype(np.float32))

    # Also save reconstruction cameras for debugging
    reconstruction_poses_path = args.out_dir / "camera_poses_reconstruction.npy"
    np.save(reconstruction_poses_path, T_world_cam_reconstruction.astype(np.float32))

    # Save predicted intrinsics used by the reconstruction branch
    intrinsics_pred_path = args.out_dir / "intrinsics_predicted.npy"
    np.save(intrinsics_pred_path, intrinsic_pred_np.astype(np.float32))

    # Save GT intrinsics mapped to the network crop geometry
    intrinsics_gt_path = args.out_dir / "intrinsics_gt_network_geometry.npy"
    intrinsics_gt_np = np.stack([g["K_net"] for g in geometries], axis=0).astype(np.float32)
    np.save(intrinsics_gt_path, intrinsics_gt_np)

    if alignment_info is not None:
        similarity_path = args.out_dir / "predicted_to_gt_similarity.npy"
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = float(alignment_info["scale"]) * np.asarray(alignment_info["rotation_matrix"], dtype=np.float32)
        S[:3, 3] = np.asarray(alignment_info["translation"], dtype=np.float32)
        np.save(similarity_path, S)

    _save_pointcloud_as_ply(ply_path, final_pts, final_cols, confidence=final_conf)
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
        "conf_thres_percentile": args.conf_thres,
        "cloud_source": args.cloud_source,
        "camera_source": args.camera_source,
        "alignment_mode": args.alignment_mode if args.camera_source == "gt" else None,
        "num_images": len(image_files),
        "num_points_output": int(len(final_pts)),
        "pointcloud_units": "meters" if args.camera_source == "gt" and args.alignment_mode == "sim3" else "arbitrary_or_model_scale",
        "output_ply": str(ply_path),
        "output_npz": str(cloud_npz_path),
        "output_camera_poses": str(poses_path),
        "output_reconstruction_camera_poses": str(reconstruction_poses_path),
        "output_predicted_intrinsics": str(intrinsics_pred_path),
        "output_gt_network_intrinsics": str(intrinsics_gt_path),
        "output_confidence_dir": str(conf_dir),
        "camera2ee_used": str(cam2ee_path) if cam2ee_path.exists() else None,
        "predicted_to_gt_similarity_path": str(similarity_path) if similarity_path is not None else None,
        "alignment_info": alignment_info,
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"[green]Saved extracted point cloud to {ply_path}[/green]")
    print(f"[green]Saved camera poses in output frame to {poses_path}[/green]")
    print(f"[green]Saved reconstruction camera poses to {reconstruction_poses_path}[/green]")
    print(f"[green]Saved predicted intrinsics to {intrinsics_pred_path}[/green]")
    print(f"[green]Saved GT network intrinsics to {intrinsics_gt_path}[/green]")
    if similarity_path is not None:
        print(f"[green]Saved predicted->GT similarity transform to {similarity_path}[/green]")
    print(f"[green]Saved confidence maps to {conf_dir}[/green]")
    print(f"[green]Saved metadata to {params_path}[/green]")

    if not args.no_wait:
        input("Data has been logged to Rerun. Open the viewer now and then press enter.")


if __name__ == "__main__":
    main()