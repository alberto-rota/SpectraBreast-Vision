"""Single-surface height-field reconstruction from a fused dense cloud.

The pipeline:

1. Project the fused cloud into a planar frame (ArUco-derived or PCA fallback).
2. Rasterize (u, v) positions into a height field with confidence-weighted
   averages; RGB and confidence are aggregated in the same pass on the GPU.
3. Iteratively fill small holes via 3x3 neighborhood voting.
4. Optionally smooth, then compute normals from finite differences on the
   height grid.
5. Triangulate observed/filled cells into a mesh and output:
     - flat arrays of surface points, colors, normals, confidence, support
     - the full HxW grids (u_coords, v_coords, height, color, normal, etc.)
     - a triangle mesh

All heavy work happens as batched tensor ops on the GPU (if available); no
explicit Python loops over points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .align import PlaneFrame


@dataclass
class SurfaceResult:
    """Outputs of `reconstruct_surface`.

    Flat arrays (one row per occupied cell):
        points: [Ns, 3], colors: [Ns, 3] uint8, normals: [Ns, 3],
        confidence: [Ns], support: [Ns], height_std: [Ns]
        triangles: [Nt, 3] int32 indexing into the flat arrays.

    Per-cell grids (HxW):
        height_grid, color_grid, xyz_grid, normal_grid, confidence_grid,
        support_grid, height_std_grid
        surface_mask (bool), observed_mask (bool), filled_mask (bool).

    Metadata:
        plane_origin [3], plane_basis [3,3] (columns = u, v, n),
        grid_step (float), grid_shape (H, W),
        u_coords [W], v_coords [H].
    """

    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    support: np.ndarray
    height_std: np.ndarray
    triangles: np.ndarray

    height_grid: np.ndarray
    color_grid: np.ndarray
    xyz_grid: np.ndarray
    normal_grid: np.ndarray
    confidence_grid: np.ndarray
    support_grid: np.ndarray
    height_std_grid: np.ndarray

    surface_mask: np.ndarray
    observed_mask: np.ndarray
    filled_mask: np.ndarray

    plane_origin: np.ndarray
    plane_basis: np.ndarray
    grid_step: float
    grid_shape: tuple[int, int]
    u_coords: np.ndarray
    v_coords: np.ndarray

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Dictionary suitable for `np.savez(...)`."""
        return {
            "points": self.points.astype(np.float32),
            "colors": self.colors.astype(np.uint8),
            "normals": self.normals.astype(np.float32),
            "confidence": self.confidence.astype(np.float32),
            "support": self.support.astype(np.float32),
            "height_std": self.height_std.astype(np.float32),
            "triangles": self.triangles.astype(np.int32),
            "height_grid": self.height_grid.astype(np.float32),
            "color_grid": self.color_grid.astype(np.uint8),
            "xyz_grid": self.xyz_grid.astype(np.float32),
            "normal_grid": self.normal_grid.astype(np.float32),
            "confidence_grid": self.confidence_grid.astype(np.float32),
            "support_grid": self.support_grid.astype(np.float32),
            "height_std_grid": self.height_std_grid.astype(np.float32),
            "surface_mask": self.surface_mask,
            "observed_mask": self.observed_mask,
            "filled_mask": self.filled_mask,
            "plane_origin": self.plane_origin.astype(np.float32),
            "plane_basis": self.plane_basis.astype(np.float32),
            "u_coords": self.u_coords.astype(np.float32),
            "v_coords": self.v_coords.astype(np.float32),
            "grid_step": np.array(self.grid_step, dtype=np.float32),
        }


def _estimate_plane_frame_from_points(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    reference_up: np.ndarray | None = None,
) -> PlaneFrame:
    """PCA-based fallback plane used when no ArUco markers are available."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N,3], got {pts.shape}")

    if weights is None:
        w = np.ones((pts.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError(f"Expected weights with shape [{pts.shape[0]}], got {w.shape}")
        w = np.clip(w, 1e-6, None)

    center = np.sum(pts * w[:, None], axis=0) / np.sum(w)
    centered = pts - center
    cov = (centered * w[:, None]).T @ centered / np.sum(w)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    normal = eigenvectors[:, 0]
    up = np.array([0.0, 0.0, 1.0]) if reference_up is None else np.asarray(reference_up, dtype=np.float64)
    if float(normal @ up) < 0.0:
        normal = -normal

    tangent_u = np.array([1.0, 0.0, 0.0])
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
    return PlaneFrame(origin=center.astype(np.float32), basis=basis)


def _transform_points_to_plane_frame(
    points: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Project world points into the plane frame: ``x_local = basis^T @ (x - origin)``.

    Works on any shape [..., 3].
    """
    pts = np.asarray(points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    return np.einsum("...j,jk->...k", pts - origin, basis)


def _estimate_grid_step(
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


def reconstruct_surface(
    fused_points: np.ndarray,            # [N,3]
    fused_colors: np.ndarray,            # [N,3] uint8
    fused_confidence: np.ndarray,        # [N]
    point_map_world: np.ndarray,         # [S,H,W,3]
    valid_masks: np.ndarray,             # [S,H,W] bool
    *,
    plane_frame: PlaneFrame | None = None,
    device: torch.device | None = None,
    grid_step: float = 0.0,
    fill_iters: int = 2,
    smooth_iters: int = 1,
    min_neighbors: int = 3,
    max_resolution: int = 2048,
) -> SurfaceResult:
    """Fit a single-surface height field on the ArUco (or fallback) plane."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fused_points = np.asarray(fused_points, dtype=np.float32)
    fused_colors = np.asarray(fused_colors, dtype=np.uint8)
    fused_confidence = np.asarray(fused_confidence, dtype=np.float32)
    point_map_world = np.asarray(point_map_world, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)

    if plane_frame is None:
        plane_weights = np.log1p(np.clip(fused_confidence, 0.0, None)) + 1e-3
        plane_frame = _estimate_plane_frame_from_points(fused_points, weights=plane_weights)

    plane_origin = np.asarray(plane_frame.origin, dtype=np.float32)
    plane_basis = np.asarray(plane_frame.basis, dtype=np.float32)

    fused_points_local = _transform_points_to_plane_frame(fused_points, plane_origin, plane_basis)  # [N,3]
    point_map_local = _transform_points_to_plane_frame(point_map_world, plane_origin, plane_basis)  # [S,H,W,3]

    if grid_step > 0.0:
        step = float(grid_step)
    else:
        step = _estimate_grid_step(
            point_map_local=point_map_local,
            valid_masks=valid_masks,
            fused_points_local=fused_points_local,
            num_views=point_map_world.shape[0],
        )

    if not np.isfinite(step) or step <= 0.0:
        uv_span = np.maximum(np.ptp(fused_points_local[:, :2], axis=0), 1e-6)
        step = float(max(uv_span.max() / 512.0, 1e-6))

    uv_min = fused_points_local[:, :2].min(axis=0)
    uv_max = fused_points_local[:, :2].max(axis=0)
    u_min = float(np.floor(uv_min[0] / step) * step)
    v_min = float(np.floor(uv_min[1] / step) * step)
    u_max = float(np.ceil(uv_max[0] / step) * step)
    v_max = float(np.ceil(uv_max[1] / step) * step)

    grid_w = int(round((u_max - u_min) / step)) + 1
    grid_h = int(round((v_max - v_min) / step)) + 1

    if max(grid_h, grid_w) > max_resolution:
        scale_factor = max(grid_h, grid_w) / float(max_resolution)
        step *= scale_factor
        u_min = float(np.floor(uv_min[0] / step) * step)
        v_min = float(np.floor(uv_min[1] / step) * step)
        u_max = float(np.ceil(uv_max[0] / step) * step)
        v_max = float(np.ceil(uv_max[1] / step) * step)
        grid_w = int(round((u_max - u_min) / step)) + 1
        grid_h = int(round((v_max - v_min) / step)) + 1

    local_pts_t = torch.as_tensor(fused_points_local, dtype=torch.float32, device=device)  # [N,3]
    colors_t = torch.as_tensor(fused_colors, dtype=torch.float32, device=device)           # [N,3]
    conf_t = torch.as_tensor(fused_confidence, dtype=torch.float32, device=device)         # [N]
    weight_t = torch.log1p(conf_t.clamp_min(0.0)) + 1e-3                                   # [N]

    u_t = local_pts_t[:, 0]
    v_t = local_pts_t[:, 1]
    h_t = local_pts_t[:, 2]

    ix = torch.clamp(torch.floor((u_t - u_min) / step).long(), 0, grid_w - 1)
    iy = torch.clamp(torch.floor((v_t - v_min) / step).long(), 0, grid_h - 1)
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

    observed_mask = (weight_sum > 0.0).view(1, 1, grid_h, grid_w)          # [1,1,H,W]
    weight_sum_safe = weight_sum.clamp_min(1e-6)
    height_grid = (height_sum / weight_sum_safe).view(1, 1, grid_h, grid_w)
    color_grid = (color_sum / weight_sum_safe[:, None]).view(1, grid_h, grid_w, 3).permute(0, 3, 1, 2)  # [1,3,H,W]
    confidence_grid = (confidence_sum / count_sum.clamp_min(1.0)).view(1, 1, grid_h, grid_w)
    support_grid = count_sum.view(1, 1, grid_h, grid_w)
    height_var_grid = (height2_sum / weight_sum_safe) - (height_sum / weight_sum_safe) ** 2
    height_std_grid = torch.sqrt(torch.clamp(height_var_grid, min=0.0)).view(1, 1, grid_h, grid_w)

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    rgb_kernel = kernel.repeat(3, 1, 1, 1)
    surface_mask = observed_mask.float()

    for _ in range(max(fill_iters, 0)):
        neighbor_count = F.conv2d(surface_mask, kernel, padding=1)
        fillable = (surface_mask == 0.0) & (neighbor_count >= float(min_neighbors))
        if not bool(torch.any(fillable)):
            break
        neighbor_safe = neighbor_count.clamp_min(1.0)
        height_avg = F.conv2d(height_grid * surface_mask, kernel, padding=1) / neighbor_safe
        color_avg = F.conv2d(color_grid * surface_mask, rgb_kernel, padding=1, groups=3) / neighbor_safe
        conf_avg = F.conv2d(confidence_grid * surface_mask, kernel, padding=1) / neighbor_safe
        std_avg = F.conv2d(height_std_grid * surface_mask, kernel, padding=1) / neighbor_safe

        height_grid = torch.where(fillable, height_avg, height_grid)
        color_grid = torch.where(fillable.expand(-1, 3, -1, -1), color_avg, color_grid)
        confidence_grid = torch.where(fillable, conf_avg, confidence_grid)
        height_std_grid = torch.where(fillable, std_avg, height_std_grid)
        surface_mask = torch.where(fillable, torch.ones_like(surface_mask), surface_mask)

    for _ in range(max(smooth_iters, 0)):
        neighbor_count = F.conv2d(surface_mask, kernel, padding=1)
        neighbor_safe = neighbor_count.clamp_min(1.0)
        height_avg = F.conv2d(height_grid * surface_mask, kernel, padding=1) / neighbor_safe
        color_avg = F.conv2d(color_grid * surface_mask, rgb_kernel, padding=1, groups=3) / neighbor_safe
        height_grid = torch.where(surface_mask > 0.0, 0.5 * height_grid + 0.5 * height_avg, height_grid)
        color_grid = torch.where(
            surface_mask.expand(-1, 3, -1, -1) > 0.0,
            0.75 * color_grid + 0.25 * color_avg,
            color_grid,
        )

    dx_kernel = torch.tensor([[[-0.5, 0.0, 0.5]]], dtype=torch.float32, device=device).unsqueeze(0) / step  # [1,1,1,3]
    dy_kernel = torch.tensor([[[-0.5], [0.0], [0.5]]], dtype=torch.float32, device=device).unsqueeze(0) / step  # [1,1,3,1]
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

    u_coords = torch.arange(grid_w, dtype=torch.float32, device=device) * step + u_min  # [W]
    v_coords = torch.arange(grid_h, dtype=torch.float32, device=device) * step + v_min  # [H]
    vv_grid, uu_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")                 # [H,W]
    local_surface_grid = torch.stack([uu_grid, vv_grid, height_grid[0, 0]], dim=-1)      # [H,W,3]

    plane_origin_t = torch.as_tensor(plane_origin, dtype=torch.float32, device=device)
    world_from_plane_t = torch.as_tensor(plane_basis.T, dtype=torch.float32, device=device)  # [3,3]
    xyz_grid = local_surface_grid.reshape(-1, 3) @ world_from_plane_t + plane_origin_t       # [H*W,3]
    normal_grid = normals_local[0].reshape(-1, 3) @ world_from_plane_t                       # [H*W,3]
    normal_grid = F.normalize(normal_grid, dim=-1).reshape(grid_h, grid_w, 3)
    xyz_grid = xyz_grid.reshape(grid_h, grid_w, 3)

    surface_mask_np = surface_mask[0, 0].detach().cpu().numpy() > 0.5
    observed_mask_np = observed_mask[0, 0].detach().cpu().numpy() > 0.5
    filled_mask_np = surface_mask_np & (~observed_mask_np)

    xyz_grid_np = xyz_grid.detach().cpu().numpy().astype(np.float32)
    color_grid_np = np.clip(color_grid[0].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 255.0).astype(np.uint8)
    normal_grid_np = normal_grid.detach().cpu().numpy().astype(np.float32)
    confidence_grid_np = confidence_grid[0, 0].detach().cpu().numpy().astype(np.float32)
    support_grid_np = support_grid[0, 0].detach().cpu().numpy().astype(np.float32)
    height_grid_np = height_grid[0, 0].detach().cpu().numpy().astype(np.float32)
    height_std_grid_np = height_std_grid[0, 0].detach().cpu().numpy().astype(np.float32)

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

    triangle_chunks: list[np.ndarray] = []
    if np.any(tri1_mask):
        triangle_chunks.append(
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
        triangle_chunks.append(
            np.stack(
                [
                    vertex_ids[:-1, 1:][tri2_mask],
                    vertex_ids[1:, 1:][tri2_mask],
                    vertex_ids[1:, :-1][tri2_mask],
                ],
                axis=1,
            )
        )
    if triangle_chunks:
        triangles_np = np.concatenate(triangle_chunks, axis=0).astype(np.int32)
    else:
        triangles_np = np.zeros((0, 3), dtype=np.int32)

    surface_points = xyz_grid_np[surface_mask_np]
    surface_colors = color_grid_np[surface_mask_np]
    surface_normals = normal_grid_np[surface_mask_np]
    surface_confidence = confidence_grid_np[surface_mask_np]
    surface_support = support_grid_np[surface_mask_np]
    surface_height_std = height_std_grid_np[surface_mask_np]

    return SurfaceResult(
        points=surface_points.astype(np.float32),
        colors=surface_colors.astype(np.uint8),
        normals=surface_normals.astype(np.float32),
        confidence=surface_confidence.astype(np.float32),
        support=surface_support.astype(np.float32),
        height_std=surface_height_std.astype(np.float32),
        triangles=triangles_np,
        height_grid=height_grid_np,
        color_grid=color_grid_np,
        xyz_grid=xyz_grid_np,
        normal_grid=normal_grid_np,
        confidence_grid=confidence_grid_np,
        support_grid=support_grid_np,
        height_std_grid=height_std_grid_np,
        surface_mask=surface_mask_np,
        observed_mask=observed_mask_np,
        filled_mask=filled_mask_np,
        plane_origin=plane_origin.astype(np.float32),
        plane_basis=plane_basis.astype(np.float32),
        grid_step=float(step),
        grid_shape=(int(grid_h), int(grid_w)),
        u_coords=u_coords.detach().cpu().numpy().astype(np.float32),
        v_coords=v_coords.detach().cpu().numpy().astype(np.float32),
    )


__all__ = ["SurfaceResult", "reconstruct_surface"]
