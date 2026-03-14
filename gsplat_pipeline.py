import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
import torch
import torch.nn.functional as F
import torchvision.io as io
from rich import print

from helpers import *
from helpers import xyzeuler_to_hmat

from gsplat.rendering import rasterization


def _parse_args():
    p = argparse.ArgumentParser(description="3DGS reconstruction + mesh extraction from posed images")

    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"), help="Directory of RGB images")
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"), help="Directory of camera pose files")
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction_3dgs"), help="Output directory")

    # 3DGS optimization
    p.add_argument("--num_gaussians", type=int, default=200_000, help="Initial number of Gaussians")
    p.add_argument("--gs_iters", type=int, default=7_000, help="3DGS optimization iterations")
    p.add_argument("--gs_log_interval", type=int, default=500, help="Log point cloud to Rerun every N iterations (0=disabled)")
    p.add_argument("--gs_batch_views", type=int, default=8, help="How many views per optimization step")
    p.add_argument("--gs_lr_xyz", type=float, default=1.6e-3, help="LR for Gaussian means (decayed over training)")
    p.add_argument("--gs_lr_features", type=float, default=2.5e-3, help="LR for Gaussian colors")
    p.add_argument("--gs_lr_opacity", type=float, default=5e-2, help="LR for Gaussian opacities")
    p.add_argument("--gs_lr_scale", type=float, default=1e-2, help="LR for Gaussian scales (decayed over training)")
    p.add_argument("--gs_lr_quat", type=float, default=1e-3, help="LR for Gaussian rotations")
    p.add_argument("--gs_lr_decay", type=float, default=0.01, help="LR decay factor for xyz/scale by end of training (1=no decay)")
    p.add_argument("--init_opacity", type=float, default=0.1, help="Initial Gaussian opacity")
    p.add_argument("--init_scale", type=float, default=0.02, help="Initial Gaussian scale")
    p.add_argument("--bg_white", action="store_true", help="Use white background instead of black")

    # Mesh extraction
    p.add_argument("--mesh_voxel_size", type=float, default=0.005, help="TSDF voxel length")
    p.add_argument("--mesh_sdf_trunc", type=float, default=0.02, help="TSDF truncation")
    p.add_argument("--mesh_depth_trunc", type=float, default=5.0, help="Max depth used in TSDF fusion")
    p.add_argument("--mesh_alpha_thr", type=float, default=0.2, help="Minimum alpha for valid depth")

    # Output / visualization
    p.add_argument("--max_points", type=int, default=2_000_000, help="Max points to save for point cloud export")
    p.add_argument("--grpc_port", type=int, default=9876, help="Rerun gRPC port")
    p.add_argument("--no_wait", action="store_true", help="Do not wait before exit")

    return p.parse_args()


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1)


def save_pointcloud_as_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save a point cloud to ASCII PLY.

    points: (N, 3)
    colors: (N, 3) uint8 or float in [0,255]
    """
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]

    pts = points.astype(np.float32)
    cols = colors
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 255).astype(np.uint8)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
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


def save_mesh_as_ply(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)


def to_o3d_intrinsics(K: np.ndarray, W: int, H: int) -> o3d.camera.PinholeCameraIntrinsic:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)


@torch.no_grad()
def estimate_scene_bounds_from_cameras(poses_c2w: torch.Tensor):
    """
    Scene bounds from camera centers only, so the init box and plot range
    match the actual pose scale (no fixed near/far that can be orders of
    magnitude wrong).
    """
    cam_centers = poses_c2w[:, :3, 3]  # [B,3]
    cam_min = cam_centers.min(dim=0).values
    cam_max = cam_centers.max(dim=0).values
    cam_center = (cam_min + cam_max) * 0.5
    cam_extent = (cam_max - cam_min).clamp(min=1e-9)
    scene_scale = cam_extent.max().item()

    # Box centered on camera cluster, tight around spread (0.6x = half-size each side)
    half = max(scene_scale * 0.6, 1e-6)
    bb_min = cam_center - half
    bb_max = cam_center + half
    return bb_min, bb_max


def initialize_gaussians_random(
    num_gaussians: int,
    poses_c2w: torch.Tensor,
    device: torch.device,
    init_scale: float,
    init_opacity: float,
):
    bb_min, bb_max = estimate_scene_bounds_from_cameras(poses_c2w)
    box_size = (bb_max - bb_min).max().item()
    # Scale Gaussians to scene: at most 3% of box so they fit and get gradient
    effective_scale = min(init_scale, max(box_size * 0.03, 1e-6))

    means = bb_min[None] + torch.rand(num_gaussians, 3, device=device) * (bb_max - bb_min)[None]
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4).repeat(num_gaussians, 1)
    log_scales = torch.full((num_gaussians, 3), np.log(effective_scale), device=device)
    opacity_logits = inverse_sigmoid(torch.full((num_gaussians,), init_opacity, device=device))
    color_logits = inverse_sigmoid(torch.rand(num_gaussians, 3, device=device).clamp(1e-4, 1.0 - 1e-4))

    gauss = {
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(quats),
        "log_scales": torch.nn.Parameter(log_scales),
        "opacity_logits": torch.nn.Parameter(opacity_logits),
        "color_logits": torch.nn.Parameter(color_logits),
    }
    return gauss


def render_views(
    gauss: dict,
    c2w: torch.Tensor,    # [B,4,4]
    Ks: torch.Tensor,     # [B,3,3]
    H: int,
    W: int,
    background: torch.Tensor,
    render_mode: str = "RGB",
):
    viewmats = torch.linalg.inv(c2w)  # world->camera

    means = gauss["means"]
    quats = normalize_quat(gauss["quats"])
    scales = torch.exp(gauss["log_scales"])
    opacities = torch.sigmoid(gauss["opacity_logits"])
    colors = torch.sigmoid(gauss["color_logits"])

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=0.01,
        far_plane=1e6,
        backgrounds=background[None].repeat(c2w.shape[0], 1),
        render_mode=render_mode,
        packed=False,
        rasterize_mode="classic",
    )
    return render_colors, render_alphas, meta


def train_3dgs(
    gauss: dict,
    images_bchw: torch.Tensor,
    poses_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    args,
):
    device = images_bchw.device
    B, _, H, W = images_bchw.shape

    gt = images_bchw.float() / 255.0
    gt = gt.permute(0, 2, 3, 1).contiguous()  # [B,H,W,3]

    background = torch.ones(3, device=device) if args.bg_white else torch.zeros(3, device=device)

    optimizer = torch.optim.Adam(
        [
            {"params": [gauss["means"]], "lr": args.gs_lr_xyz},
            {"params": [gauss["color_logits"]], "lr": args.gs_lr_features},
            {"params": [gauss["opacity_logits"]], "lr": args.gs_lr_opacity},
            {"params": [gauss["log_scales"]], "lr": args.gs_lr_scale},
            {"params": [gauss["quats"]], "lr": args.gs_lr_quat},
        ]
    )

    for it in range(args.gs_iters):
        # Exponential LR decay for xyz and scale (refine toward end)
        decay = args.gs_lr_decay ** (min(it, args.gs_iters - 1) / max(args.gs_iters - 1, 1))
        optimizer.param_groups[0]["lr"] = args.gs_lr_xyz * decay
        optimizer.param_groups[3]["lr"] = args.gs_lr_scale * decay

        idx = torch.randperm(B, device=device)[: min(args.gs_batch_views, B)]

        pred_rgb, pred_alpha, _ = render_views(
            gauss=gauss,
            c2w=poses_c2w[idx],
            Ks=intrinsics[idx],
            H=H,
            W=W,
            background=background,
            render_mode="RGB",
        )

        target = gt[idx]

        rgb_loss = F.l1_loss(pred_rgb, target)

        # Light regularization
        opacity_reg = 1e-4 * torch.sigmoid(gauss["opacity_logits"]).mean()
        scale_reg = 1e-4 * torch.exp(gauss["log_scales"]).mean()

        loss = rgb_loss + opacity_reg + scale_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            gauss["quats"].data = normalize_quat(gauss["quats"].data)
            gauss["log_scales"].data.clamp_(min=np.log(1e-5), max=np.log(0.2))

        # Log point cloud to Rerun every gs_log_interval iterations
        if args.gs_log_interval > 0 and (it % args.gs_log_interval == 0 or it == args.gs_iters - 1):
            pts_np = gauss["means"].detach().cpu().numpy()
            cols_np = (torch.sigmoid(gauss["color_logits"]).detach().cpu().numpy() * 255.0).astype(np.uint8)
            if len(pts_np) > args.max_points:
                idx_pts = np.random.choice(len(pts_np), args.max_points, replace=False)
                pts_np, cols_np = pts_np[idx_pts], cols_np[idx_pts]
            rr.set_time("iteration", sequence=it)
            rr.log("/points", rr.Points3D(pts_np, colors=cols_np))

        if it % 100 == 0 or it == args.gs_iters - 1:
            print(
                f"[cyan][3DGS][/cyan] iter={it:04d} "
                f"loss={loss.item():.6f} "
                f"rgb={rgb_loss.item():.6f} "
                f"n_gauss={gauss['means'].shape[0]}"
            )

    return gauss


@torch.no_grad()
def extract_mesh_tsdf(
    gauss: dict,
    images_bchw: torch.Tensor,
    poses_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    args,
):
    device = images_bchw.device
    B, _, H, W = images_bchw.shape
    background = torch.ones(3, device=device) if args.bg_white else torch.zeros(3, device=device)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.mesh_voxel_size,
        sdf_trunc=args.mesh_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(B):
        render_out, alpha_out, _ = render_views(
            gauss=gauss,
            c2w=poses_c2w[i:i + 1],
            Ks=intrinsics[i:i + 1],
            H=H,
            W=W,
            background=background,
            render_mode="RGB+ED",  # RGB + expected depth
        )

        rgb = render_out[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
        depth = render_out[0, ..., 3].cpu().numpy()
        alpha = alpha_out[0, ..., 0].cpu().numpy()

        depth = np.where(alpha >= args.mesh_alpha_thr, depth, 0.0)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth = np.where(depth > 0.0, depth, 0.0)
        depth = np.where(depth < args.mesh_depth_trunc, depth, 0.0)

        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        depth_f32 = depth.astype(np.float32)

        color_o3d = o3d.geometry.Image(rgb_u8)
        depth_o3d = o3d.geometry.Image(depth_f32)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=args.mesh_depth_trunc,
            convert_rgb_to_intensity=False,
        )

        K = intrinsics[i].detach().cpu().numpy()
        intrinsic_o3d = to_o3d_intrinsics(K, W, H)

        extrinsic = torch.linalg.inv(poses_c2w[i]).detach().cpu().numpy().astype(np.float64)
        volume.integrate(rgbd, intrinsic_o3d, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _log_mesh_to_rerun(mesh: o3d.geometry.TriangleMesh, path: str = "/mesh") -> None:
    """Convert Open3D mesh to Rerun Mesh3D and log."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    mesh3d_kw = {
        "vertex_positions": verts,
        "triangle_indices": tris,
        "vertex_normals": normals,
    }
    if mesh.has_vertex_colors():
        vcols = (np.asarray(mesh.vertex_colors, dtype=np.float32) * 255.0).astype(np.uint8)
        mesh3d_kw["vertex_colors"] = vcols
    rr.log(path, rr.Mesh3D(**mesh3d_kw))


def main():
    args = _parse_args()

    device = torch.device("cuda")
    print("Using device:", f"[green]{device}[/green]" if device.type == "cuda" else f"[red]{device}[/red]")

    rgb_image_dir = args.rgb_dir
    camera_pose_dir = args.pose_dir
    image_files = sorted(rgb_image_dir.glob("image_*.png"))
    pose_files = sorted(camera_pose_dir.glob("pose_*.txt"))

    # Load images as [C, H, W] tensors to GPU
    images = [io.read_image(str(p)).to(device) for p in image_files]

    # Load poses as [6] tensors to GPU
    poses = [
        torch.tensor(
            [float(x) for x in p.read_text().strip().split()],
            dtype=torch.float32,
            device=device,
        )
        for p in pose_files
    ]

    if not images or not poses:
        raise RuntimeError("No images or poses found.")

    if len(images) != len(poses):
        raise RuntimeError(f"Mismatch: found {len(images)} images but {len(poses)} poses")

    # Batch everything
    images_tensor = torch.stack(images)
    poses_tensor = torch.stack(poses)
    print(f"Loaded {len(images)} pairs.")
    print(f"Batched images shape: {images_tensor.shape}")
    print(f"Batched poses6 shape: {poses_tensor.shape}")

    translation_scale = 1
    poses_tensor4 = xyzeuler_to_hmat(
        poses_tensor, convention="ROLLPITCHYAW", translation_scale=translation_scale
    )
    print(f"Batched poses shape: {poses_tensor4.shape}")

    # Load intrinsics and cam2ee, expand as batch
    intrinsics = np.load("camera_parameters/intrinsics.npy")
    intrinsics = (
        torch.tensor(intrinsics, dtype=torch.float32)
        .to(device)
        .unsqueeze(0)
        .repeat(len(poses_tensor), 1, 1)
    )

    cam2ee = np.load("camera_parameters/camera2ee.npy")

    # Fix cam2ee if needed
    if cam2ee.shape == (3, 4):
        cam2ee_fixed = np.eye(4, dtype=np.float32)
        cam2ee_fixed[:3, :4] = cam2ee
        cam2ee = cam2ee_fixed

    cam2ee = torch.tensor(cam2ee, dtype=torch.float32).to(device).unsqueeze(0).repeat(len(poses_tensor), 1, 1)
    poses_tensor_prem = torch.matmul(poses_tensor4, cam2ee)  # [N,4,4], camera->world

    print(f"Batched intrinsic shape: {intrinsics.shape}")

    rr.init("SpectraBreast_3DGS")
    rr.serve_grpc(grpc_port=args.grpc_port)

    # 3DGS init (bounds from camera spread so plot range matches pose scale)
    bb_min, bb_max = estimate_scene_bounds_from_cameras(poses_tensor_prem)
    extent = (bb_max - bb_min).cpu().numpy()
    print(f"Scene bounds from cameras: extent ≈ [{extent[0]:.6f}, {extent[1]:.6f}, {extent[2]:.6f}]")
    print("Initializing Gaussians...")
    gauss = initialize_gaussians_random(
        num_gaussians=args.num_gaussians,
        poses_c2w=poses_tensor_prem,
        device=device,
        init_scale=args.init_scale,
        init_opacity=args.init_opacity,
    )

    # 3DGS train
    print("Training 3DGS...")
    gauss = train_3dgs(
        gauss=gauss,
        images_bchw=images_tensor,
        poses_c2w=poses_tensor_prem,
        intrinsics=intrinsics,
        args=args,
    )

    # Final point cloud from Gaussian centers (for saving and Rerun)
    final_pts = gauss["means"].detach().cpu().numpy()
    final_colors = (torch.sigmoid(gauss["color_logits"]).detach().cpu().numpy() * 255.0).astype(np.uint8)
    if len(final_pts) > args.max_points:
        indices = np.random.choice(len(final_pts), args.max_points, replace=False)
        final_pts = final_pts[indices]
        final_colors = final_colors[indices]

    # Log final point cloud and cameras to Rerun
    rr.set_time("iteration", sequence=args.gs_iters)
    rr.log("/points", rr.Points3D(final_pts, colors=final_colors))
    print("Logging to Rerun...")
    for i in range(len(poses_tensor_prem)):
        pose = poses_tensor_prem[i].detach().cpu().numpy()  # [4,4] camera->world
        K = intrinsics[i].detach().cpu().numpy()
        H, W = images_tensor.shape[-2], images_tensor.shape[-1]

        rr.log(
            f"/cameras/{i}",
            rr.Transform3D(
                translation=pose[:3, 3],
                mat3x3=pose[:3, :3],
            ),
        )
        rr.log(
            f"/cameras/{i}/image",
            rr.Pinhole(
                image_from_camera=K,
                resolution=[W, H],
                image_plane_distance=0.01
            ),
        )

    # Extract mesh once at the end
    print("Extracting mesh from 3DGS rendered depth...")
    mesh = extract_mesh_tsdf(
        gauss=gauss,
        images_bchw=images_tensor,
        poses_c2w=poses_tensor_prem,
        intrinsics=intrinsics,
        args=args,
    )

    # Log final mesh to Rerun
    rr.set_time("iteration", sequence=args.gs_iters)
    _log_mesh_to_rerun(mesh, path="/mesh")

    # Save outputs
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pointcloud_path = out_dir / "gaussians_centers.ply"
    mesh_path = out_dir / "mesh_3dgs_tsdf.ply"

    print(f"Saving point cloud to {pointcloud_path}...")
    save_pointcloud_as_ply(pointcloud_path, final_pts, final_colors)

    print(f"Saving mesh to {mesh_path}...")
    save_mesh_as_ply(mesh_path, mesh)
    input("Logged to Rerun. Press Enter when the viewer has been launched...")
    # Save parameters
    params = {
        "rgb_dir": str(args.rgb_dir),
        "pose_dir": str(args.pose_dir),
        "out_dir": str(args.out_dir),
        "num_images": int(images_tensor.shape[0]),
        "image_shape": list(images_tensor.shape),
        "poses6_shape": list(poses_tensor.shape),
        "poses4_shape": list(poses_tensor4.shape),
        "intrinsics_shape": list(intrinsics.shape),

        "num_gaussians": args.num_gaussians,
        "gs_iters": args.gs_iters,
        "gs_log_interval": args.gs_log_interval,
        "gs_batch_views": args.gs_batch_views,
        "gs_lr_xyz": args.gs_lr_xyz,
        "gs_lr_features": args.gs_lr_features,
        "gs_lr_opacity": args.gs_lr_opacity,
        "gs_lr_scale": args.gs_lr_scale,
        "gs_lr_quat": args.gs_lr_quat,
        "gs_lr_decay": args.gs_lr_decay,
        "init_opacity": args.init_opacity,
        "init_scale": args.init_scale,
        "bg_white": args.bg_white,

        "mesh_voxel_size": args.mesh_voxel_size,
        "mesh_sdf_trunc": args.mesh_sdf_trunc,
        "mesh_depth_trunc": args.mesh_depth_trunc,
        "mesh_alpha_thr": args.mesh_alpha_thr,

        "num_points_output": int(len(final_pts)),
        "output_pointcloud": str(pointcloud_path),
        "output_mesh": str(mesh_path),
    }

    params_path = out_dir / "reconstruction_3dgs_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Parameters saved to {params_path}")

    if not args.no_wait:
        input("Data has been logged to rerun. Open the viewer now and then press enter")

    print("[green]3DGS reconstruction + mesh extraction complete![/green]")


if __name__ == "__main__":
    main()