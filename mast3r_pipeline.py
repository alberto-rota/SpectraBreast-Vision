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


def _parse_args():
    p = argparse.ArgumentParser(
        description="Calibrated dense fusion with MASt3R correspondences + exact robot poses"
    )
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction"))

    # MASt3R
    p.add_argument("--model_name", type=str,
                   default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--neighbor_window", type=int, default=2,
                   help="Use temporal neighbors i-k...i-1 and i+1...i+k")

    # Matching
    p.add_argument("--desc_conf_thr", type=float, default=0.2,
                   help="Descriptor confidence threshold")
    p.add_argument("--pixel_tol", type=float, default=1.5,
                   help="Optional subpixel tolerance placeholder")
    p.add_argument("--max_matches_per_pair", type=int, default=50000)
    p.add_argument("--min_reproj_px", type=float, default=2.0,
                   help="Max average reprojection error in pixels")
    p.add_argument("--min_triang_angle_deg", type=float, default=1.0,
                   help="Minimum triangulation angle")

    # Fusion
    p.add_argument("--voxel_size", type=float, default=0.0015)
    p.add_argument("--max_points", type=int, default=2_000_000)

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


def _save_pointcloud_as_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    assert points.shape[0] == colors.shape[0]
    N = points.shape[0]

    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    data = np.concatenate([pts, cols.astype(np.float32)], axis=1)
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv)

    out_pts = np.zeros((len(uniq), 3), dtype=np.float64)
    out_cols = np.zeros((len(uniq), 3), dtype=np.float64)

    for d in range(3):
        out_pts[:, d] = np.bincount(inv, weights=points[:, d], minlength=len(uniq)) / np.maximum(counts, 1)
        out_cols[:, d] = np.bincount(inv, weights=colors[:, d], minlength=len(uniq)) / np.maximum(counts, 1)

    return out_pts.astype(np.float32), np.clip(out_cols, 0, 255).astype(np.uint8)


def _make_projection(K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """
    Returns P = K [R_cw | t_cw] mapping world -> image.
    T_world_cam is camera->world.
    """
    T_cam_world = np.linalg.inv(T_world_cam)
    return K @ T_cam_world[:3, :]


def _triangulate_points(P1, P2, uv1, uv2):
    """
    uv1, uv2: Nx2 pixel coordinates
    returns Nx3 points in world coordinates
    """
    pts4d = cv2.triangulatePoints(
        P1.astype(np.float64),
        P2.astype(np.float64),
        uv1.T.astype(np.float64),
        uv2.T.astype(np.float64),
    )
    pts3d = (pts4d[:3] / pts4d[3:4]).T
    return pts3d.astype(np.float32)


def _project(P, pts3d_world):
    pts_h = np.concatenate([pts3d_world, np.ones((len(pts3d_world), 1), dtype=np.float32)], axis=1)
    q = (P @ pts_h.T).T
    uv = q[:, :2] / q[:, 2:3]
    return uv


def _camera_center(T_world_cam: np.ndarray) -> np.ndarray:
    return T_world_cam[:3, 3]


def _triangulation_angle_deg(C1, C2, X):
    v1 = X - C1[None, :]
    v2 = X - C2[None, :]
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
    cosang = np.sum(v1 * v2, axis=1).clip(-1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang


def _positive_depth_mask(T_world_cam: np.ndarray, pts3d_world: np.ndarray):
    T_cam_world = np.linalg.inv(T_world_cam)
    pts_h = np.concatenate([pts3d_world, np.ones((len(pts3d_world), 1), dtype=np.float32)], axis=1)
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]
    return pts_cam[:, 2] > 1e-6


@torch.no_grad()
def _match_pair(
    img_i,
    img_j,
    model,
    device,
    desc_conf_thr: float,
    max_matches_per_pair: int,
):
    """
    Returns:
        uv_i: Nx2 in pixels of image i
        uv_j: Nx2 in pixels of image j
        conf: Nx1 match confidence proxy
    """
    from dust3r.inference import inference

    # Depending on checkout/version, one of these imports should exist.
    # Keep both attempts for robustness.
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs
    except Exception:
        from mast3r.utils.misc import fast_reciprocal_NNs

    output = inference([(img_i, img_j)], model, device, batch_size=1, verbose=False)
    pred1 = output["pred1"]
    pred2 = output["pred2"]

    desc1 = pred1["desc"].squeeze(0)          # HxWxD
    desc2 = pred2["desc"].squeeze(0)          # HxWxD
    desc_conf1 = pred1["desc_conf"].squeeze(0)
    desc_conf2 = pred2["desc_conf"].squeeze(0)

    if desc_conf1.ndim == 3:
        desc_conf1 = desc_conf1[..., 0]
    if desc_conf2.ndim == 3:
        desc_conf2 = desc_conf2[..., 0]

    valid1 = desc_conf1 >= desc_conf_thr
    valid2 = desc_conf2 >= desc_conf_thr

    H1, W1, D = desc1.shape
    H2, W2, _ = desc2.shape

    y1, x1 = torch.where(valid1)
    y2, x2 = torch.where(valid2)

    if len(x1) == 0 or len(x2) == 0:
        return None, None, None

    # fast_reciprocal_NNs expects full descriptor maps (H,W,D) and optional (x,y) seed pixels
    x1_np = x1.cpu().numpy()
    y1_np = y1.cpu().numpy()
    xy1, xy2 = fast_reciprocal_NNs(
        desc1.float(),
        desc2.float(),
        subsample_or_initxy1=(x1_np, y1_np),
        ret_xy=True,
        device=device,
        dist="dot",
        block_size=2**13,
    )

    if xy1 is None or len(xy1) == 0:
        return None, None, None

    # xy1, xy2 are (N, 2) in (x, y) order from merge_corres
    uv_i = xy1.astype(np.float32)
    uv_j = xy2.astype(np.float32)

    r1 = torch.as_tensor(xy1[:, 1], device=desc_conf1.device, dtype=torch.long)
    c1 = torch.as_tensor(xy1[:, 0], device=desc_conf1.device, dtype=torch.long)
    r2 = torch.as_tensor(xy2[:, 1], device=desc_conf2.device, dtype=torch.long)
    c2 = torch.as_tensor(xy2[:, 0], device=desc_conf2.device, dtype=torch.long)
    conf = np.minimum(
        desc_conf1[r1, c1].cpu().numpy().astype(np.float32),
        desc_conf2[r2, c2].cpu().numpy().astype(np.float32),
    )

    if len(uv_i) > max_matches_per_pair:
        topk = np.argsort(conf)[-max_matches_per_pair:][::-1]
        uv_i = uv_i[topk]
        uv_j = uv_j[topk]
        conf = conf[topk]

    return uv_i, uv_j, conf


def main():
    args = _parse_args()

    device = torch.device("cuda")
    print("Using device:", f"[green]{device}[/green]")

    image_files = sorted(args.rgb_dir.glob("image_*.png"))
    pose_files = sorted(args.pose_dir.glob("pose_*.txt"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.rgb_dir}")
    if len(image_files) != len(pose_files):
        raise RuntimeError(f"Found {len(image_files)} images but {len(pose_files)} pose files")

    # Load original RGB for colors/logging
    orig_images = [io.read_image(str(p)) for p in image_files]  # [3,H,W], uint8, CPU
    H0, W0 = int(orig_images[0].shape[1]), int(orig_images[0].shape[2])

    # Load robot poses
    poses6 = []
    for p in pose_files:
        vals = [float(x) for x in p.read_text().strip().split()]
        poses6.append(vals)
    poses6 = torch.tensor(poses6, dtype=torch.float32, device=device)

    T_world_ee = xyzeuler_to_hmat(
        poses6, convention="ROLLPITCHYAW", translation_scale=1.0
    )  # [N,4,4]

    K = np.load(args.camera_params_dir / "intrinsics.npy").astype(np.float32)
    T_ee_cam = _fix_3x4_to_4x4(np.load(args.camera_params_dir / "camera2ee.npy"))

    # Same convention as your original script
    T_ee_cam_t = torch.tensor(T_ee_cam, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(poses6), 1, 1)
    T_world_cam = torch.matmul(T_world_ee, T_ee_cam_t).detach().cpu().numpy()

    rr.init("SpectraBreast_CalibratedDenseTriangulation")
    rr.serve_grpc(grpc_port=args.grpc_port)

    sys.path.append("/home/arota/spectra/mast3r")
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.image import load_images

    print(f"Loading MASt3R model: {args.model_name}")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)
    model.eval()

    filelist = [str(p) for p in image_files]
    imgs_mast3r = load_images(filelist, size=args.image_size, verbose=True)

    # Projection matrices with exact calibration
    Ps = [_make_projection(K, T_world_cam[i]) for i in range(len(image_files))]
    Cs = [_camera_center(T_world_cam[i]) for i in range(len(image_files))]

    all_pts = []
    all_cols = []

    n = len(image_files)
    for i in range(n):
        for j in range(max(0, i - args.neighbor_window), min(n, i + args.neighbor_window + 1)):
            if j <= i:
                continue

            print(f"[cyan]Matching pair ({i}, {j})[/cyan]")

            uv_i, uv_j, mconf = _match_pair(
                imgs_mast3r[i],
                imgs_mast3r[j],
                model=model,
                device=device,
                desc_conf_thr=args.desc_conf_thr,
                max_matches_per_pair=args.max_matches_per_pair,
            )

            if uv_i is None or len(uv_i) == 0:
                print(f"[yellow]No matches for pair ({i}, {j})[/yellow]")
                continue

            # IMPORTANT:
            # load_images() may resize internally for the network.
            # We scale matched coordinates back to the original RGB resolution.
            Hi_net, Wi_net = imgs_mast3r[i]["img"].shape[-2:]
            Hj_net, Wj_net = imgs_mast3r[j]["img"].shape[-2:]

            sx_i, sy_i = W0 / float(Wi_net), H0 / float(Hi_net)
            sx_j, sy_j = W0 / float(Wj_net), H0 / float(Hj_net)

            uv_i[:, 0] *= sx_i
            uv_i[:, 1] *= sy_i
            uv_j[:, 0] *= sx_j
            uv_j[:, 1] *= sy_j

            pts3d = _triangulate_points(Ps[i], Ps[j], uv_i, uv_j)

            # Positive depth in both views
            pos_i = _positive_depth_mask(T_world_cam[i], pts3d)
            pos_j = _positive_depth_mask(T_world_cam[j], pts3d)

            # Triangulation angle
            ang = _triangulation_angle_deg(Cs[i], Cs[j], pts3d)
            good_ang = ang >= args.min_triang_angle_deg

            # Reprojection error
            uv_i_hat = _project(Ps[i], pts3d)
            uv_j_hat = _project(Ps[j], pts3d)
            err_i = np.linalg.norm(uv_i_hat - uv_i, axis=1)
            err_j = np.linalg.norm(uv_j_hat - uv_j, axis=1)
            good_err = 0.5 * (err_i + err_j) <= args.min_reproj_px

            keep = pos_i & pos_j & good_ang & good_err & np.isfinite(pts3d).all(axis=1)
            pts3d = pts3d[keep]
            uv_i_keep = uv_i[keep]

            if len(pts3d) == 0:
                print(f"[yellow]Pair ({i}, {j}): all points filtered[/yellow]")
                continue

            # Color from image i
            rgb_i = orig_images[i].permute(1, 2, 0).cpu().numpy()
            u = np.clip(np.round(uv_i_keep[:, 0]).astype(np.int32), 0, W0 - 1)
            v = np.clip(np.round(uv_i_keep[:, 1]).astype(np.int32), 0, H0 - 1)
            cols = rgb_i[v, u].astype(np.uint8)

            all_pts.append(pts3d.astype(np.float32))
            all_cols.append(cols)

            print(f"[green]Pair ({i}, {j}): kept {len(pts3d):,} triangulated points[/green]")

    if len(all_pts) == 0:
        raise RuntimeError("No valid triangulated points were produced")

    final_pts = np.concatenate(all_pts, axis=0)
    final_cols = np.concatenate(all_cols, axis=0)

    print(f"Raw triangulated cloud: {len(final_pts):,} points")

    if args.voxel_size > 0:
        final_pts, final_cols = _voxel_downsample(final_pts, final_cols, args.voxel_size)
        print(f"After voxel downsample: {len(final_pts):,} points")

    if len(final_pts) > args.max_points:
        idx = np.random.choice(len(final_pts), args.max_points, replace=False)
        final_pts = final_pts[idx]
        final_cols = final_cols[idx]

    # Log calibrated cameras
    for i in range(len(image_files)):
        T = T_world_cam[i]
        rr.log(
            f"/cameras/{i}",
            rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        )
        rr.log(
            f"/cameras/{i}/image",
            rr.Pinhole(image_from_camera=K, resolution=[W0, H0]),
        )

    rr.log("/points", rr.Points3D(final_pts, colors=final_cols))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = args.out_dir / "mast3r_calibrated_triangulated.ply"
    _save_pointcloud_as_ply(ply_path, final_pts, final_cols)

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
        "min_reproj_px": args.min_reproj_px,
        "min_triang_angle_deg": args.min_triang_angle_deg,
        "voxel_size": args.voxel_size,
        "max_points": args.max_points,
        "num_images": len(image_files),
        "num_points_output": int(len(final_pts)),
        "output_ply": str(ply_path),
    }
    with open(args.out_dir / "mast3r_calibrated_triangulated_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"[green]Saved point cloud to {ply_path}[/green]")

    if not args.no_wait:
        input("Data has been logged to Rerun. Open the viewer now and then press enter.")


if __name__ == "__main__":
    main()