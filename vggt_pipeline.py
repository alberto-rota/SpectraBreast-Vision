import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io
from rich import print

import rerun as rr
from helpers import xyzeuler_to_hmat
import rerun.blueprint as rrb


def _send_rerun_blueprint() -> None:
    """Single 3D view for the only logged entity: ``/points``."""
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(name="Point cloud", origin="/", contents=["/points"]),
            )
        ),
        make_active=True,
    )


def _parse_args():
    p = argparse.ArgumentParser(description="VGGT Point Cloud Extraction")
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"))
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"))
    p.add_argument("--camera_params_dir", type=Path, default=Path("camera_parameters"))
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reconstruction_vggt"),
        help="Root directory where timestamped reconstruction subfolders are created",
    )
    p.add_argument(
        "--cad_mesh_path",
        type=Path,
        default=Path("sample_cad.stl"),
        help="Optional path to an ASCII STL CAD mesh to overlay in all 3D views",
    )

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
    p.add_argument(
        "--surface_grid_step",
        type=float,
        default=0.0,
        help="Height-field grid step in output units. <=0 estimates it automatically.",
    )
    p.add_argument(
        "--surface_fill_iters",
        type=int,
        default=2,
        help="Neighborhood hole-filling passes applied to the surface grid",
    )
    p.add_argument(
        "--surface_smooth_iters",
        type=int,
        default=1,
        help="Neighborhood smoothing passes applied to the surface grid",
    )
    p.add_argument(
        "--surface_min_neighbors",
        type=int,
        default=3,
        help="Minimum valid neighbors required before an empty surface cell is filled",
    )
    p.add_argument(
        "--surface_max_resolution",
        type=int,
        default=2048,
        help="Maximum number of cells along the longest side of the surface grid",
    )

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
    normals: np.ndarray | None = None,
    support: np.ndarray | None = None,
) -> None:
    assert points.shape[0] == colors.shape[0]
    if confidence is not None:
        assert confidence.shape[0] == points.shape[0]
    if normals is not None:
        assert normals.shape == points.shape
    if support is not None:
        assert support.shape[0] == points.shape[0]

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
    ]
    if normals is not None:
        header.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    header.extend(
        [
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        ]
    )
    if confidence is not None:
        header.append("property float confidence")
    if support is not None:
        header.append("property float support")
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        data_parts = [pts]
        fmt = ["%.6f", "%.6f", "%.6f"]
        if normals is not None:
            data_parts.append(normals.astype(np.float32))
            fmt.extend(["%.6f", "%.6f", "%.6f"])
        data_parts.append(cols.astype(np.float32))
        fmt.extend(["%d", "%d", "%d"])
        if confidence is not None:
            data_parts.append(confidence.reshape(-1, 1).astype(np.float32))
            fmt.append("%.6f")
        if support is not None:
            data_parts.append(support.reshape(-1, 1).astype(np.float32))
            fmt.append("%.6f")
        data = np.concatenate(data_parts, axis=1)
        np.savetxt(f, data, fmt=fmt)


def _save_mesh_as_ply(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> None:
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape [N,3], got {verts.shape}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"Expected triangles with shape [M,3], got {tris.shape}")
    if vertex_colors is not None and vertex_colors.shape != verts.shape:
        raise ValueError(f"Expected vertex_colors with shape {verts.shape}, got {vertex_colors.shape}")
    if vertex_normals is not None and vertex_normals.shape != verts.shape:
        raise ValueError(f"Expected vertex_normals with shape {verts.shape}, got {vertex_normals.shape}")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {verts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if vertex_normals is not None:
        header.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    if vertex_colors is not None:
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header.extend(
        [
            f"element face {tris.shape[0]}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    )

    vertex_parts = [verts]
    vertex_fmt = ["%.6f", "%.6f", "%.6f"]
    if vertex_normals is not None:
        vertex_parts.append(vertex_normals.astype(np.float32))
        vertex_fmt.extend(["%.6f", "%.6f", "%.6f"])
    if vertex_colors is not None:
        vertex_parts.append(vertex_colors.astype(np.float32))
        vertex_fmt.extend(["%d", "%d", "%d"])
    vertex_data = np.concatenate(vertex_parts, axis=1)

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if len(vertex_data) > 0:
            np.savetxt(f, vertex_data, fmt=vertex_fmt)
        if len(tris) > 0:
            face_data = np.concatenate(
                [np.full((tris.shape[0], 1), 3, dtype=np.int32), tris.astype(np.int32)],
                axis=1,
            )
            np.savetxt(f, face_data, fmt="%d %d %d %d")


def _load_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal ASCII STL loader.
    Returns:
        vertices: [Nv,3] float32
        triangles: [Nt,3] int32 (indices into vertices)
    """
    vertices_raw: list[tuple[float, float, float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.lower().startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            vertices_raw.append((x, y, z))

    if not vertices_raw:
        raise ValueError(f"No vertices found in STL file {path}")
    if len(vertices_raw) % 3 != 0:
        raise ValueError(f"Vertex count in STL file {path} is not a multiple of 3")

    tri_vertices = np.asarray(vertices_raw, dtype=np.float32).reshape(-1, 3, 3)  # [Nt,3,3]

    # Deduplicate vertices to keep triangles reasonably sized.
    unique_verts_list: list[tuple[float, float, float]] = []
    index_map: dict[tuple[float, float, float], int] = {}
    triangles_list: list[tuple[int, int, int]] = []

    for tri in tri_vertices:
        tri_idx: list[int] = []
        for v in tri:
            key = (float(v[0]), float(v[1]), float(v[2]))
            if key not in index_map:
                index_map[key] = len(unique_verts_list)
                unique_verts_list.append(key)
            tri_idx.append(index_map[key])
        triangles_list.append((tri_idx[0], tri_idx[1], tri_idx[2]))

    vertices = np.asarray(unique_verts_list, dtype=np.float32)
    triangles = np.asarray(triangles_list, dtype=np.int32)
    return vertices, triangles


def _prepare_output_run_dir(base_dir: Path) -> tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_dir.name


def _update_most_recent_link(base_dir: Path, run_dir: Path) -> Path:
    most_recent_dir = base_dir / "most-recent"
    if most_recent_dir.is_symlink() or most_recent_dir.is_file():
        most_recent_dir.unlink()
    elif most_recent_dir.exists():
        shutil.rmtree(most_recent_dir)

    try:
        most_recent_dir.symlink_to(run_dir.relative_to(base_dir), target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, most_recent_dir)

    return most_recent_dir


def _log_mesh_to_rerun(
    path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> None:
    mesh3d_kwargs = {
        "vertex_positions": np.asarray(vertices, dtype=np.float32),
        "triangle_indices": np.asarray(triangles, dtype=np.uint32),
    }
    if vertex_colors is not None:
        mesh3d_kwargs["vertex_colors"] = np.asarray(vertex_colors, dtype=np.uint8)
    if vertex_normals is not None:
        mesh3d_kwargs["vertex_normals"] = np.asarray(vertex_normals, dtype=np.float32)
    rr.log(path, rr.Mesh3D(**mesh3d_kwargs), static=True)


def _estimate_surface_plane_frame(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    reference_up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N,3], got {pts.shape}")

    if weights is None:
        w = np.ones((pts.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError(f"Expected weights with shape [{pts.shape[0]}], got {w.shape}")
        w = np.clip(w, a_min=1e-6, a_max=None)

    center = np.sum(pts * w[:, None], axis=0) / np.sum(w)
    centered = pts - center
    cov = (centered * w[:, None]).T @ centered / np.sum(w)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    normal = eigenvectors[:, 0]
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if reference_up is None else np.asarray(reference_up, dtype=np.float64)
    if float(normal @ up) < 0.0:
        normal = -normal

    tangent_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_u = tangent_u - normal * float(tangent_u @ normal)
    tangent_u_norm = np.linalg.norm(tangent_u)
    if tangent_u_norm < 1e-6:
        tangent_u = eigenvectors[:, 2]
    else:
        tangent_u = tangent_u / tangent_u_norm

    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(np.linalg.norm(tangent_v), 1e-12)
    tangent_u = np.cross(tangent_v, normal)
    tangent_u = tangent_u / max(np.linalg.norm(tangent_u), 1e-12)

    basis = np.stack([tangent_u, tangent_v, normal], axis=1).astype(np.float32)
    return center.astype(np.float32), basis, eigenvalues.astype(np.float32)


def _transform_points_to_plane_frame(points: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    return np.einsum("...j,jk->...k", pts - origin, basis)


def _estimate_surface_grid_step(
    point_map_local: np.ndarray,
    valid_masks: np.ndarray,
    fused_points_local: np.ndarray,
    num_views: int,
) -> float:
    uv = point_map_local[..., :2]  # [S,H,W,2]

    du = np.linalg.norm(uv[:, :, 1:, :] - uv[:, :, :-1, :], axis=-1)
    valid_du = valid_masks[:, :, 1:] & valid_masks[:, :, :-1]

    dv = np.linalg.norm(uv[:, 1:, :, :] - uv[:, :-1, :, :], axis=-1)
    valid_dv = valid_masks[:, 1:, :] & valid_masks[:, :-1, :]

    spacings = np.concatenate([du[valid_du], dv[valid_dv]], axis=0)
    spacings = spacings[np.isfinite(spacings) & (spacings > 1e-8)]
    if spacings.size > 0:
        return float(np.median(spacings))

    uv_points = fused_points_local[:, :2]
    span = np.maximum(np.ptp(uv_points, axis=0), 1e-6)
    area = float(span[0] * span[1])
    effective_points = max(float(fused_points_local.shape[0]) / max(num_views, 1), 1.0)
    return float(np.sqrt(area / effective_points))


def _reconstruct_single_surface(
    fused_points: np.ndarray,
    fused_colors: np.ndarray,
    fused_confidence: np.ndarray,
    point_map_world: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
    requested_grid_step: float,
    fill_iters: int,
    smooth_iters: int,
    min_neighbors: int,
    max_resolution: int,
) -> dict[str, np.ndarray | float | int | tuple[int, int]]:
    plane_weights = np.log1p(np.clip(fused_confidence.astype(np.float32), a_min=0.0, a_max=None)) + 1e-3
    plane_origin, plane_basis, plane_eigenvalues = _estimate_surface_plane_frame(
        fused_points,
        weights=plane_weights,
    )

    fused_points_local = _transform_points_to_plane_frame(fused_points, plane_origin, plane_basis)  # [N,3]
    point_map_local = _transform_points_to_plane_frame(point_map_world, plane_origin, plane_basis)  # [S,H,W,3]

    if requested_grid_step > 0.0:
        grid_step = float(requested_grid_step)
    else:
        grid_step = _estimate_surface_grid_step(
            point_map_local=point_map_local,
            valid_masks=valid_masks,
            fused_points_local=fused_points_local,
            num_views=point_map_world.shape[0],
        )

    if not np.isfinite(grid_step) or grid_step <= 0.0:
        uv_span = np.maximum(np.ptp(fused_points_local[:, :2], axis=0), 1e-6)
        grid_step = float(max(uv_span.max() / 512.0, 1e-6))

    uv_min = fused_points_local[:, :2].min(axis=0)
    uv_max = fused_points_local[:, :2].max(axis=0)
    u_min = float(np.floor(uv_min[0] / grid_step) * grid_step)
    v_min = float(np.floor(uv_min[1] / grid_step) * grid_step)
    u_max = float(np.ceil(uv_max[0] / grid_step) * grid_step)
    v_max = float(np.ceil(uv_max[1] / grid_step) * grid_step)

    grid_w = int(round((u_max - u_min) / grid_step)) + 1
    grid_h = int(round((v_max - v_min) / grid_step)) + 1

    if max(grid_h, grid_w) > max_resolution:
        scale = max(grid_h, grid_w) / float(max_resolution)
        grid_step *= scale
        u_min = float(np.floor(uv_min[0] / grid_step) * grid_step)
        v_min = float(np.floor(uv_min[1] / grid_step) * grid_step)
        u_max = float(np.ceil(uv_max[0] / grid_step) * grid_step)
        v_max = float(np.ceil(uv_max[1] / grid_step) * grid_step)
        grid_w = int(round((u_max - u_min) / grid_step)) + 1
        grid_h = int(round((v_max - v_min) / grid_step)) + 1

    local_pts_t = torch.as_tensor(fused_points_local, dtype=torch.float32, device=device)  # [N,3]
    colors_t = torch.as_tensor(fused_colors, dtype=torch.float32, device=device)  # [N,3]
    conf_t = torch.as_tensor(fused_confidence, dtype=torch.float32, device=device)  # [N]
    weight_t = torch.log1p(conf_t.clamp_min(0.0)) + 1e-3  # [N]

    u_t = local_pts_t[:, 0]
    v_t = local_pts_t[:, 1]
    h_t = local_pts_t[:, 2]

    ix = torch.clamp(torch.floor((u_t - u_min) / grid_step).long(), min=0, max=grid_w - 1)
    iy = torch.clamp(torch.floor((v_t - v_min) / grid_step).long(), min=0, max=grid_h - 1)
    flat_idx = iy * grid_w + ix  # [N]
    num_cells = grid_h * grid_w

    weight_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    height_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    height2_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    confidence_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    count_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    color_sum = torch.zeros((num_cells, 3), dtype=torch.float32, device=device)

    weight_sum.scatter_add_(0, flat_idx, weight_t)
    height_sum.scatter_add_(0, flat_idx, weight_t * h_t)
    height2_sum.scatter_add_(0, flat_idx, weight_t * h_t * h_t)
    confidence_sum.scatter_add_(0, flat_idx, conf_t)
    count_sum.scatter_add_(0, flat_idx, torch.ones_like(conf_t))
    color_sum.scatter_add_(0, flat_idx[:, None].expand(-1, 3), weight_t[:, None] * colors_t)

    observed_mask = (weight_sum > 0.0).view(1, 1, grid_h, grid_w)  # [1,1,H,W]
    weight_sum_safe = weight_sum.clamp_min(1e-6)

    height_grid = (height_sum / weight_sum_safe).view(1, 1, grid_h, grid_w)  # [1,1,H,W]
    color_grid = (color_sum / weight_sum_safe[:, None]).view(1, grid_h, grid_w, 3).permute(0, 3, 1, 2)  # [1,3,H,W]
    confidence_grid = (confidence_sum / count_sum.clamp_min(1.0)).view(1, 1, grid_h, grid_w)  # [1,1,H,W]
    support_grid = count_sum.view(1, 1, grid_h, grid_w)  # [1,1,H,W]
    height_var_grid = (height2_sum / weight_sum_safe) - (height_sum / weight_sum_safe) ** 2
    height_std_grid = torch.sqrt(torch.clamp(height_var_grid, min=0.0)).view(1, 1, grid_h, grid_w)  # [1,1,H,W]

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    rgb_kernel = kernel.repeat(3, 1, 1, 1)
    surface_mask = observed_mask.float()

    for _ in range(max(fill_iters, 0)):
        neighbor_count = F.conv2d(surface_mask, kernel, padding=1)
        fillable = (surface_mask == 0.0) & (neighbor_count >= float(min_neighbors))
        if not bool(torch.any(fillable)):
            break

        height_avg = F.conv2d(height_grid * surface_mask, kernel, padding=1) / neighbor_count.clamp_min(1.0)
        color_avg = F.conv2d(color_grid * surface_mask, rgb_kernel, padding=1, groups=3) / neighbor_count.clamp_min(1.0)
        conf_avg = F.conv2d(confidence_grid * surface_mask, kernel, padding=1) / neighbor_count.clamp_min(1.0)
        std_avg = F.conv2d(height_std_grid * surface_mask, kernel, padding=1) / neighbor_count.clamp_min(1.0)

        height_grid = torch.where(fillable, height_avg, height_grid)
        color_grid = torch.where(fillable.expand(-1, 3, -1, -1), color_avg, color_grid)
        confidence_grid = torch.where(fillable, conf_avg, confidence_grid)
        height_std_grid = torch.where(fillable, std_avg, height_std_grid)
        surface_mask = torch.where(fillable, torch.ones_like(surface_mask), surface_mask)

    for _ in range(max(smooth_iters, 0)):
        neighbor_count = F.conv2d(surface_mask, kernel, padding=1)
        height_avg = F.conv2d(height_grid * surface_mask, kernel, padding=1) / neighbor_count.clamp_min(1.0)
        color_avg = F.conv2d(color_grid * surface_mask, rgb_kernel, padding=1, groups=3) / neighbor_count.clamp_min(1.0)
        height_grid = torch.where(surface_mask > 0.0, 0.5 * height_grid + 0.5 * height_avg, height_grid)
        color_grid = torch.where(
            surface_mask.expand(-1, 3, -1, -1) > 0.0,
            0.75 * color_grid + 0.25 * color_avg,
            color_grid,
        )

    dx_kernel = torch.tensor([[[-0.5, 0.0, 0.5]]], dtype=torch.float32, device=device).unsqueeze(0) / grid_step  # [1,1,1,3]
    dy_kernel = torch.tensor([[[-0.5], [0.0], [0.5]]], dtype=torch.float32, device=device).unsqueeze(0) / grid_step  # [1,1,3,1]
    grad_u = F.conv2d(F.pad(height_grid, (1, 1, 0, 0), mode="replicate"), dx_kernel)
    grad_v = F.conv2d(F.pad(height_grid, (0, 0, 1, 1), mode="replicate"), dy_kernel)

    normals_local = torch.stack(
        [
            -grad_u[:, 0],
            -grad_v[:, 0],
            torch.ones((1, grid_h, grid_w), dtype=torch.float32, device=device),
        ],
        dim=-1,
    )  # [1,H,W,3]
    normals_local = F.normalize(normals_local, dim=-1)
    normals_local = torch.where(surface_mask.permute(0, 2, 3, 1) > 0.0, normals_local, torch.zeros_like(normals_local))

    u_coords = torch.arange(grid_w, dtype=torch.float32, device=device) * grid_step + u_min  # [W]
    v_coords = torch.arange(grid_h, dtype=torch.float32, device=device) * grid_step + v_min  # [H]
    vv_grid, uu_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")  # [H,W]
    local_surface_grid = torch.stack(
        [uu_grid, vv_grid, height_grid[0, 0]],
        dim=-1,
    )  # [H,W,3]

    plane_origin_t = torch.as_tensor(plane_origin, dtype=torch.float32, device=device)
    world_from_plane_t = torch.as_tensor(plane_basis.T, dtype=torch.float32, device=device)  # [3,3]
    xyz_grid = local_surface_grid.reshape(-1, 3) @ world_from_plane_t + plane_origin_t  # [H*W,3]
    normal_grid = normals_local[0].reshape(-1, 3) @ world_from_plane_t  # [H*W,3]
    normal_grid = F.normalize(normal_grid, dim=-1).reshape(grid_h, grid_w, 3)
    xyz_grid = xyz_grid.reshape(grid_h, grid_w, 3)

    surface_mask_np = surface_mask[0, 0].detach().cpu().numpy() > 0.5
    observed_mask_np = observed_mask[0, 0].detach().cpu().numpy()
    observed_mask_np = observed_mask_np > 0.5
    filled_mask_np = surface_mask_np & (~observed_mask_np)

    xyz_grid_np = xyz_grid.detach().cpu().numpy().astype(np.float32)  # [H,W,3]
    color_grid_np = np.clip(color_grid[0].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 255.0).astype(np.uint8)  # [H,W,3]
    normal_grid_np = normal_grid.detach().cpu().numpy().astype(np.float32)  # [H,W,3]
    confidence_grid_np = confidence_grid[0, 0].detach().cpu().numpy().astype(np.float32)  # [H,W]
    support_grid_np = support_grid[0, 0].detach().cpu().numpy().astype(np.float32)  # [H,W]
    height_grid_np = height_grid[0, 0].detach().cpu().numpy().astype(np.float32)  # [H,W]
    height_std_grid_np = height_std_grid[0, 0].detach().cpu().numpy().astype(np.float32)  # [H,W]

    vertex_ids = -np.ones((grid_h, grid_w), dtype=np.int32)
    vertex_ids[surface_mask_np] = np.arange(int(surface_mask_np.sum()), dtype=np.int32)

    m00 = surface_mask_np[:-1, :-1]
    m10 = surface_mask_np[:-1, 1:]
    m01 = surface_mask_np[1:, :-1]
    m11 = surface_mask_np[1:, 1:]

    o00 = observed_mask_np[:-1, :-1].astype(np.int32)
    o10 = observed_mask_np[:-1, 1:].astype(np.int32)
    o01 = observed_mask_np[1:, :-1].astype(np.int32)
    o11 = observed_mask_np[1:, 1:].astype(np.int32)

    tri1_mask = m00 & m10 & m01 & ((o00 + o10 + o01) >= 2)
    tri2_mask = m10 & m11 & m01 & ((o10 + o11 + o01) >= 2)

    triangles = []
    if np.any(tri1_mask):
        triangles.append(
            np.stack(
                [
                    vertex_ids[:-1, :-1][tri1_mask],
                    vertex_ids[:-1, 1:][tri1_mask],
                    vertex_ids[1:, :-1][tri1_mask],
                ],
                axis=1,
            )
        )
    if np.any(tri2_mask):
        triangles.append(
            np.stack(
                [
                    vertex_ids[:-1, 1:][tri2_mask],
                    vertex_ids[1:, 1:][tri2_mask],
                    vertex_ids[1:, :-1][tri2_mask],
                ],
                axis=1,
            )
        )
    triangles_np = np.concatenate(triangles, axis=0).astype(np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)

    surface_points = xyz_grid_np[surface_mask_np]  # [Ns,3]
    surface_colors = color_grid_np[surface_mask_np]  # [Ns,3]
    surface_normals = normal_grid_np[surface_mask_np]  # [Ns,3]
    surface_confidence = confidence_grid_np[surface_mask_np]  # [Ns]
    surface_support = support_grid_np[surface_mask_np]  # [Ns]
    surface_height_std = height_std_grid_np[surface_mask_np]  # [Ns]

    return {
        "plane_origin": plane_origin.astype(np.float32),
        "plane_basis": plane_basis.astype(np.float32),
        "plane_eigenvalues": plane_eigenvalues.astype(np.float32),
        "grid_step": float(grid_step),
        "grid_shape": (int(grid_h), int(grid_w)),
        "u_coords": u_coords.detach().cpu().numpy().astype(np.float32),
        "v_coords": v_coords.detach().cpu().numpy().astype(np.float32),
        "height_grid": height_grid_np,
        "color_grid": color_grid_np,
        "xyz_grid": xyz_grid_np,
        "normal_grid": normal_grid_np,
        "confidence_grid": confidence_grid_np,
        "support_grid": support_grid_np,
        "height_std_grid": height_std_grid_np,
        "surface_mask": surface_mask_np,
        "observed_mask": observed_mask_np,
        "filled_mask": filled_mask_np,
        "points": surface_points.astype(np.float32),
        "colors": surface_colors.astype(np.uint8),
        "normals": surface_normals.astype(np.float32),
        "confidence": surface_confidence.astype(np.float32),
        "support": surface_support.astype(np.float32),
        "height_std": surface_height_std.astype(np.float32),
        "triangles": triangles_np,
    }


def _compute_cad_height_error(
    surface: dict[str, np.ndarray | float | int | tuple[int, int]],
    cad_vertices_world: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the CAD mesh into the same planar frame and grid as the reconstructed
    surface, estimate a per-cell CAD height, and derive a per-cell and per-point
    height error (surface minus CAD) along the plane normal.
    Returns:
        cad_height_grid: [H,W] float32
        height_error_grid: [H,W] float32
        height_error_per_point: [Ns] float32 (aligned with surface['points'])
    """
    if cad_vertices_world.size == 0:
        raise ValueError("CAD mesh has no vertices")

    plane_origin = np.asarray(surface["plane_origin"], dtype=np.float32)  # [3]
    plane_basis = np.asarray(surface["plane_basis"], dtype=np.float32)  # [3,3]
    grid_h, grid_w = surface["grid_shape"]  # tuple[int,int]
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    grid_step = float(surface["grid_step"])
    u_coords = np.asarray(surface["u_coords"], dtype=np.float32)  # [W]
    v_coords = np.asarray(surface["v_coords"], dtype=np.float32)  # [H]
    u0 = float(u_coords[0])
    v0 = float(v_coords[0])

    cad_local = _transform_points_to_plane_frame(cad_vertices_world, plane_origin, plane_basis)  # [Nc,3]

    cad_local_t = torch.as_tensor(cad_local, dtype=torch.float32, device=device)  # [Nc,3]
    u_c = cad_local_t[:, 0]
    v_c = cad_local_t[:, 1]
    h_c = cad_local_t[:, 2]

    ix = torch.clamp(torch.floor((u_c - u0) / grid_step).long(), min=0, max=grid_w - 1)
    iy = torch.clamp(torch.floor((v_c - v0) / grid_step).long(), min=0, max=grid_h - 1)
    flat_idx = iy * grid_w + ix  # [Nc]
    num_cells = grid_h * grid_w

    cad_height_sum = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    cad_count = torch.zeros((num_cells,), dtype=torch.float32, device=device)

    cad_height_sum.scatter_add_(0, flat_idx, h_c)
    cad_count.scatter_add_(0, flat_idx, torch.ones_like(h_c))

    cad_count_safe = cad_count.clamp_min(1.0)
    cad_height_grid_t = (cad_height_sum / cad_count_safe).view(grid_h, grid_w)  # [H,W]
    cad_mask_t = (cad_count > 0.0).view(grid_h, grid_w)  # [H,W]

    cad_height_grid = cad_height_grid_t.detach().cpu().numpy().astype(np.float32)
    cad_mask = cad_mask_t.detach().cpu().numpy()

    surface_height_grid = np.asarray(surface["height_grid"], dtype=np.float32)  # [H,W]
    surface_mask = np.asarray(surface["surface_mask"], dtype=bool)  # [H,W]

    height_error_grid = np.zeros_like(surface_height_grid, dtype=np.float32)
    valid = surface_mask & cad_mask
    height_error_grid[valid] = surface_height_grid[valid] - cad_height_grid[valid]

    error_flat = height_error_grid.reshape(-1)  # [H*W]
    surface_mask_flat = surface_mask.reshape(-1)  # [H*W]
    height_error_per_point = error_flat[surface_mask_flat]  # [Ns]

    return cad_height_grid, height_error_grid, height_error_per_point


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


def _rerun_flush() -> None:
    """Push batched Rerun data to the gRPC sink so the viewer can display it."""
    try:
        flush_fn = getattr(rr, "flush", None)
        if callable(flush_fn):
            flush_fn()
            return
        for getter_name in ("get_data_recording", "get_global_data_recording"):
            getter = getattr(rr, getter_name, None)
            if not callable(getter):
                continue
            rec = getter()
            if rec is not None and hasattr(rec, "flush"):
                rec.flush()
                return
    except Exception as exc:
        print(
            f"[yellow]Rerun flush failed ({exc}); try reconnecting the viewer.[/yellow]"
        )


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

    _log_step("Step 1/8 - Loading RGB images, poses, and calibration")
    image_files = sorted(
        list(args.rgb_dir.glob("*.png")) +
        list(args.rgb_dir.glob("*.jpg")) +
        list(args.rgb_dir.glob("*.jpeg"))
    )
    pose_files = sorted(args.pose_dir.glob("pose_*.txt")) if args.pose_dir.exists() else []
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.rgb_dir}")

    # GT poses are optional - if missing/mismatched, run in RGB-only mode.
    have_gt_poses = len(pose_files) == len(image_files)
    if len(pose_files) > 0 and not have_gt_poses:
        print(
            f"[yellow]Found {len(image_files)} images but {len(pose_files)} pose files; "
            "ignoring pose files and running in RGB-only mode.[/yellow]"
        )
    elif len(pose_files) == 0:
        print("[yellow]No pose files found; running in RGB-only mode (no GT cameras).[/yellow]")

    orig_images = [io.read_image(str(path)) for path in image_files]  # [3,H,W], uint8

    if have_gt_poses:
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
    else:
        poses6 = None
        T_world_ee = None

    # GT intrinsics are optional - if missing, we rely on VGGT predicted intrinsics.
    intrinsics_path = args.camera_params_dir / "intrinsics.npy"
    cam2ee_path = args.camera_params_dir / "camera2ee.npy"
    have_gt_intrinsics = intrinsics_path.exists()
    if have_gt_intrinsics:
        K_orig = np.load(intrinsics_path).astype(np.float32)
    else:
        K_orig = None
        print(
            f"[yellow]{intrinsics_path} not found; will use VGGT predicted intrinsics "
            "for visualization.[/yellow]"
        )

    if have_gt_poses:
        if cam2ee_path.exists():
            T_ee_cam = _fix_3x4_to_4x4(np.load(cam2ee_path))
            print(f"Loaded camera-to-EE transform from [green]{cam2ee_path}[/green]")
        else:
            T_ee_cam = np.eye(4, dtype=np.float32)
            print("[yellow]camera2ee.npy not found, assuming poses are already camera poses.[/yellow]")

        T_ee_cam_t = torch.tensor(T_ee_cam, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(poses6), 1, 1)
        T_world_cam_gt = torch.matmul(T_world_ee, T_ee_cam_t).detach().cpu().numpy().astype(np.float32)
    else:
        T_world_cam_gt = None

    if args.camera_source == "gt" and not have_gt_poses:
        print(
            "[yellow]--camera_source gt was requested but no GT poses are available; "
            "falling back to --camera_source predicted.[/yellow]"
        )
        args.camera_source = "predicted"

    print(
        f"Loaded {len(image_files)} images"
        + (f" and {len(pose_files)} poses" if have_gt_poses else " (RGB-only)")
    )
    cad_mesh_vertices = None
    cad_mesh_triangles = None
    if args.cad_mesh_path is not None:
        if args.cad_mesh_path.exists():
            try:
                _log_step(f"Loading CAD mesh from {args.cad_mesh_path}")
                cad_mesh_vertices, cad_mesh_triangles = _load_ascii_stl(args.cad_mesh_path)
                print(
                    f"[green]Loaded CAD mesh[/green]\n"
                    f"  vertices: {cad_mesh_vertices.shape[0]:,}\n"
                    f"  triangles: {cad_mesh_triangles.shape[0]:,}"
                )
            except Exception as exc:
                print(
                    "[yellow]Failed to load CAD mesh from "
                    f"{args.cad_mesh_path}: {exc}[/yellow]"
                )
        else:
            print(f"[yellow]CAD mesh path {args.cad_mesh_path} does not exist; skipping.[/yellow]")
    run_out_dir, run_timestamp = _prepare_output_run_dir(args.out_dir)
    most_recent_dir = args.out_dir / "most-recent"
    print(f"Saving this reconstruction under [green]{run_out_dir}[/green]")

    _log_step("Step 2/8 - Initializing Rerun and loading the VGGT model")
    try:
        rr.disconnect()
        print("[green]Disconnected from Previous Rerun Session[/green]")
    except Exception as exc:
        print(
            "[yellow]Could not disconnect from previous Rerun session "
            f"({exc}). Continuing anyway.[/yellow]"
        )
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
    # Keep inference numerics and model weights in reduced precision on CUDA to
    # cut persistent VRAM usage without changing image resolution or call flow.
    model = VGGT.from_pretrained(args.model_name).to(device=device, dtype=dtype)
    model.eval()

    _log_step("Step 3/8 - Preparing VGGT image geometry")
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

        if have_gt_intrinsics:
            K_net = np.eye(3, dtype=np.float32)
            K_net[0, 0] = K_orig[0, 0] * scale_x
            K_net[1, 1] = K_orig[1, 1] * scale_y
            K_net[0, 2] = K_orig[0, 2] * scale_x
            K_net[1, 2] = K_orig[1, 2] * scale_y - start_y
        else:
            # Placeholder; filled with VGGT-predicted intrinsics after inference.
            K_net = None

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

    import time
    start_time = time.time()
    _log_step(
        f"Step 4/8 - Running VGGT inference "
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

    # If no GT intrinsics were provided, populate K_net from the VGGT-predicted
    # intrinsics (already in network pixel coordinates, shape [S,3,3]).
    if not have_gt_intrinsics:
        for idx, geom in enumerate(geometries):
            geom["K_net"] = intrinsic_pred_np[idx].astype(np.float32)

    alignment_info = None
    similarity_path = None

    _log_step("Step 5/8 - Aligning reconstruction frame if requested")
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

    _log_step("Step 6/8 - Consolidating valid 3D points")
    all_pts = []
    all_cols = []
    all_conf = []
    valid_masks = []
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
        valid_masks.append(valid.copy())

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

    _log_step("Step 7/8 - Reconstructing a single-surface height field")
    valid_masks_np = np.stack(valid_masks, axis=0)
    surface = _reconstruct_single_surface(
        fused_points=final_pts,
        fused_colors=final_cols,
        fused_confidence=final_conf,
        point_map_world=point_map_in_output_frame,
        valid_masks=valid_masks_np,
        device=device,
        requested_grid_step=args.surface_grid_step,
        fill_iters=args.surface_fill_iters,
        smooth_iters=args.surface_smooth_iters,
        min_neighbors=args.surface_min_neighbors,
        max_resolution=args.surface_max_resolution,
    )
    if surface["points"].shape[0] == 0:
        raise RuntimeError("Surface reconstruction produced no valid points")

    print(f"Surface reconstruction time: {time.time() - start_time:.2f} seconds")
    cad_height_grid = None
    height_error_grid = None
    height_error_per_point = None
    if cad_mesh_vertices is not None and cad_mesh_triangles is not None:
        try:
            _log_step("Computing CAD height field and per-point height error")
            cad_height_grid, height_error_grid, height_error_per_point = _compute_cad_height_error(
                surface=surface,
                cad_vertices_world=cad_mesh_vertices,
                device=device,
            )
            print(
                f"[green]CAD height error computed[/green]\n"
                f"  mean |error|: {float(np.mean(np.abs(height_error_per_point))):.6f}\n"
                f"  max |error| : {float(np.max(np.abs(height_error_per_point))):.6f}"
            )
        except Exception as exc:
            print(
                "[yellow]Failed to compute CAD height error; proceeding without it: "
                f"{exc}[/yellow]"
            )

    print(
        f"[green]Surface reconstruction complete[/green]\n"
        f"  grid: {surface['grid_shape'][0]} x {surface['grid_shape'][1]}\n"
        f"  grid step: {float(surface['grid_step']):.6f}\n"
        f"  surface points: {surface['points'].shape[0]:,}\n"
        f"  mesh triangles: {surface['triangles'].shape[0]:,}\n"
        f"  interpolated cells: {int(surface['filled_mask'].sum()):,}"
    )

    _log_step("Step 8/8 - Logging point cloud to Rerun and saving the reconstruction")
    conf_dir = run_out_dir / "confidence_maps"
    conf_dir.mkdir(parents=True, exist_ok=True)

    # Rerun: single dense point cloud (RGB from VGGT input crop), nothing else.
    rr.log("/points", rr.Points3D(final_pts, colors=final_cols), static=True)
    _rerun_flush()
    _send_rerun_blueprint()
    _rerun_flush()

    for image_idx, conf_vis in enumerate(conf_visualizations):
        cv2.imwrite(
            str(conf_dir / f"confidence_{image_idx:03d}.png"),
            cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR),
        )

    ply_path = run_out_dir / "vggt_extracted_cloud.ply"
    cloud_npz_path = run_out_dir / "vggt_extracted_cloud.npz"
    surface_ply_path = run_out_dir / "vggt_surface_cloud.ply"
    surface_npz_path = run_out_dir / "vggt_surface_cloud.npz"
    surface_mesh_path = run_out_dir / "vggt_surface_mesh.ply"
    surface_grid_path = run_out_dir / "vggt_surface_height_field.npz"
    params_path = run_out_dir / "vggt_extracted_cloud_params.json"

    # Canonical output camera poses:
    # - predicted mode -> save predicted reconstruction cameras
    # - gt mode -> save GT camera poses, because the final cloud is intended to be in the GT/world frame
    poses_path = run_out_dir / "camera_poses_output_frame.npy"
    if args.camera_source == "gt":
        np.save(poses_path, T_world_cam_gt.astype(np.float32))
    else:
        np.save(poses_path, T_world_cam_pred.astype(np.float32))

    # Also save reconstruction cameras for debugging
    reconstruction_poses_path = run_out_dir / "camera_poses_reconstruction.npy"
    np.save(reconstruction_poses_path, T_world_cam_reconstruction.astype(np.float32))

    # Save predicted intrinsics used by the reconstruction branch
    intrinsics_pred_path = run_out_dir / "intrinsics_predicted.npy"
    np.save(intrinsics_pred_path, intrinsic_pred_np.astype(np.float32))

    # Save GT intrinsics mapped to the network crop geometry (only if GT was provided).
    intrinsics_gt_path: Path | None = None
    if have_gt_intrinsics:
        intrinsics_gt_path = run_out_dir / "intrinsics_gt_network_geometry.npy"
        intrinsics_gt_np = np.stack([g["K_net"] for g in geometries], axis=0).astype(np.float32)
        np.save(intrinsics_gt_path, intrinsics_gt_np)

    if alignment_info is not None:
        similarity_path = run_out_dir / "predicted_to_gt_similarity.npy"
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = float(alignment_info["scale"]) * np.asarray(alignment_info["rotation_matrix"], dtype=np.float32)
        S[:3, 3] = np.asarray(alignment_info["translation"], dtype=np.float32)
        np.save(similarity_path, S)

    _save_pointcloud_as_ply(ply_path, final_pts, final_cols, confidence=final_conf)
    _save_pointcloud_as_ply(
        surface_ply_path,
        surface["points"],
        surface["colors"],
        confidence=surface["confidence"],
        normals=surface["normals"],
        support=surface["support"],
    )
    _save_mesh_as_ply(
        surface_mesh_path,
        vertices=surface["points"],
        triangles=surface["triangles"],
        vertex_colors=surface["colors"],
        vertex_normals=surface["normals"],
    )
    np.savez(
        cloud_npz_path,
        points=final_pts.astype(np.float32),
        colors=final_cols.astype(np.uint8),
        confidence=final_conf.astype(np.float32),
    )
    surface_npz_kwargs = {
        "points": surface["points"].astype(np.float32),
        "colors": surface["colors"].astype(np.uint8),
        "normals": surface["normals"].astype(np.float32),
        "confidence": surface["confidence"].astype(np.float32),
        "support": surface["support"].astype(np.float32),
        "height_std": surface["height_std"].astype(np.float32),
        "triangles": surface["triangles"].astype(np.int32),
    }
    if height_error_per_point is not None:
        surface_npz_kwargs["height_error"] = height_error_per_point.astype(np.float32)
    np.savez(surface_npz_path, **surface_npz_kwargs)
    surface_grid_npz_kwargs = {
        "plane_origin": surface["plane_origin"].astype(np.float32),
        "plane_basis": surface["plane_basis"].astype(np.float32),
        "plane_eigenvalues": surface["plane_eigenvalues"].astype(np.float32),
        "grid_step": np.array(surface["grid_step"], dtype=np.float32),
        "u_coords": surface["u_coords"].astype(np.float32),
        "v_coords": surface["v_coords"].astype(np.float32),
        "height_grid": surface["height_grid"].astype(np.float32),
        "color_grid": surface["color_grid"].astype(np.uint8),
        "xyz_grid": surface["xyz_grid"].astype(np.float32),
        "normal_grid": surface["normal_grid"].astype(np.float32),
        "confidence_grid": surface["confidence_grid"].astype(np.float32),
        "support_grid": surface["support_grid"].astype(np.float32),
        "height_std_grid": surface["height_std_grid"].astype(np.float32),
        "surface_mask": surface["surface_mask"],
        "observed_mask": surface["observed_mask"],
        "filled_mask": surface["filled_mask"],
    }
    if cad_height_grid is not None:
        surface_grid_npz_kwargs["cad_height_grid"] = cad_height_grid.astype(np.float32)
    if height_error_grid is not None:
        surface_grid_npz_kwargs["height_error_grid"] = height_error_grid.astype(np.float32)
    np.savez(surface_grid_path, **surface_grid_npz_kwargs)

    n_surf = int(surface["points"].shape[0])
    normal_arrow_stride = max(n_surf // 12000, 1)
    normal_arrow_count = (n_surf + normal_arrow_stride - 1) // normal_arrow_stride if n_surf > 0 else 0

    params = {
        "rgb_dir": str(args.rgb_dir),
        "pose_dir": str(args.pose_dir),
        "camera_params_dir": str(args.camera_params_dir),
        "out_dir_root": str(args.out_dir),
        "run_output_dir": str(run_out_dir),
        "run_timestamp": run_timestamp,
        "most_recent_dir": str(most_recent_dir),
        "model_name": args.model_name,
        "image_size": args.image_size,
        "conf_thres_percentile": args.conf_thres,
        "cloud_source": args.cloud_source,
        "camera_source": args.camera_source,
        "alignment_mode": args.alignment_mode if args.camera_source == "gt" else None,
        "num_images": len(image_files),
        "num_points_output": int(len(final_pts)),
        "num_surface_points_output": int(surface["points"].shape[0]),
        "num_surface_mesh_triangles": int(surface["triangles"].shape[0]),
        "pointcloud_units": "meters" if args.camera_source == "gt" and args.alignment_mode == "sim3" else "arbitrary_or_model_scale",
        "output_ply": str(ply_path),
        "output_npz": str(cloud_npz_path),
        "output_surface_ply": str(surface_ply_path),
        "output_surface_npz": str(surface_npz_path),
        "output_surface_mesh": str(surface_mesh_path),
        "output_surface_grid": str(surface_grid_path),
        "output_camera_poses": str(poses_path),
        "output_reconstruction_camera_poses": str(reconstruction_poses_path),
        "output_predicted_intrinsics": str(intrinsics_pred_path),
        "output_gt_network_intrinsics": str(intrinsics_gt_path) if intrinsics_gt_path is not None else None,
        "output_confidence_dir": str(conf_dir),
        "have_gt_poses": bool(have_gt_poses),
        "have_gt_intrinsics": bool(have_gt_intrinsics),
        "camera2ee_used": str(cam2ee_path) if (have_gt_poses and cam2ee_path.exists()) else None,
        "predicted_to_gt_similarity_path": str(similarity_path) if similarity_path is not None else None,
        "alignment_info": alignment_info,
        "surface_reconstruction": {
            "method": "dominant-plane confidence-weighted height field",
            "grid_step": float(surface["grid_step"]),
            "grid_shape": [int(surface["grid_shape"][0]), int(surface["grid_shape"][1])],
            "plane_origin": surface["plane_origin"].tolist(),
            "plane_basis": surface["plane_basis"].tolist(),
            "plane_eigenvalues": surface["plane_eigenvalues"].tolist(),
            "surface_grid_step_requested": args.surface_grid_step,
            "surface_fill_iters": args.surface_fill_iters,
            "surface_smooth_iters": args.surface_smooth_iters,
            "surface_min_neighbors": args.surface_min_neighbors,
            "surface_max_resolution": args.surface_max_resolution,
            "num_surface_observed_cells": int(surface["observed_mask"].sum()),
            "num_surface_filled_cells": int(surface["filled_mask"].sum()),
            "normal_arrow_stride": int(normal_arrow_stride),
            "normal_arrow_count": int(normal_arrow_count),
        },
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    _update_most_recent_link(args.out_dir, run_out_dir)

    print(f"[green]Saved extracted point cloud to {ply_path}[/green]")
    print(f"[green]Saved single-surface point cloud to {surface_ply_path}[/green]")
    print(f"[green]Saved single-surface mesh to {surface_mesh_path}[/green]")
    print(f"[green]Saved single-surface height field grid to {surface_grid_path}[/green]")
    print(f"[green]Saved camera poses in output frame to {poses_path}[/green]")
    print(f"[green]Saved reconstruction camera poses to {reconstruction_poses_path}[/green]")
    print(f"[green]Saved predicted intrinsics to {intrinsics_pred_path}[/green]")
    if intrinsics_gt_path is not None:
        print(f"[green]Saved GT network intrinsics to {intrinsics_gt_path}[/green]")
    if similarity_path is not None:
        print(f"[green]Saved predicted->GT similarity transform to {similarity_path}[/green]")
    print(f"[green]Saved confidence maps to {conf_dir}[/green]")
    print(f"[green]Saved metadata to {params_path}[/green]")
    print(f"[green]Updated most-recent output at {most_recent_dir}[/green]")

    if not args.no_wait:
        input("Data has been logged to Rerun. Open the viewer now and then press enter.")


if __name__ == "__main__":
    main()