"""Output-frame conventions. World :math:`Z` (third axis) is inverted when
``z_axis_points_down`` is set so that **up** in a typical "camera + gravity"
sense is **:math:`-Z`** (Z increases toward the ground) — common for robotics
and for matching some viewer defaults when combined with a scene flip.

Reflection :math:`S = \\mathrm{diag}(1,1,-1)`:  :math:`p' = (x, y, -z)`.
For rigid/se3, :math:`T' = S  T`. For 3D basis (columns in world), :math:`B' = S  B`."""

from __future__ import annotations

import numpy as np

from .surface import SurfaceResult

_S3: np.ndarray = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
_S4: np.ndarray = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)


def reflect_z_points_inplace(pts: np.ndarray) -> None:
    # (N,3), (H,W,3), (S,H,W,3) — (..., 3)
    if pts.size == 0 or pts.shape[-1] < 3:
        return
    pts[..., 2] *= -1.0


def reflect_z_T_world_cam_inplace(t_world: np.ndarray) -> None:
    # (4,4) or (V,4,4), world from camera: T' = S4 @ T
    if t_world.size == 0 or t_world.shape[-2:] != (4, 4):
        return
    t_world[...] = np.einsum("ab,...bc->...ac", _S4, t_world)


def reflect_z_marker_corners_dict_inplace(by_id: dict[int, np.ndarray]) -> None:
    for a in by_id.values():
        if a is not None and a.size and a.shape[-1] == 3:
            a[..., 2] *= -1.0


def reflect_z_sim3_matrix(sim3: np.ndarray) -> np.ndarray:
    s = np.asarray(sim3, dtype=np.float32).reshape(4, 4)
    return (_S4 @ s).astype(np.float32, copy=False)


def apply_z_flip_to_surface_result(surface: SurfaceResult) -> None:
    """Mutate a surface in place so it matches world Z reflected (S @ world)."""
    for name in ("points", "normals", "xyz_grid", "normal_grid"):
        arr = getattr(surface, name, None)
        if isinstance(arr, np.ndarray) and arr.size and arr.shape[-1] == 3:
            reflect_z_points_inplace(arr)
    h2d = surface.height_grid
    if isinstance(h2d, np.ndarray) and h2d.size:
        h2d[...] *= -1.0
    pbo = surface.plane_origin
    if isinstance(pbo, np.ndarray) and pbo.shape == (3,):
        pbo[2] *= -1.0
    pbb = surface.plane_basis
    if isinstance(pbb, np.ndarray) and pbb.shape == (3, 3):
        pbb[:, :] = (_S3 @ pbb).astype(np.float32, copy=False)


def apply_output_z_reflection(
    *,
    aligned_cloud: np.ndarray,
    aligned_poses: np.ndarray,
    aligned_markers_in_output_frame: dict[int, np.ndarray] | None,
    surface: SurfaceResult,
) -> None:
    """In-place: negate world :math:`Z` on the dense cloud, poses, markers, and surface."""
    if aligned_cloud is not None and aligned_cloud.size:
        reflect_z_points_inplace(aligned_cloud)
    if aligned_poses is not None and aligned_poses.size:
        reflect_z_T_world_cam_inplace(aligned_poses)
    if aligned_markers_in_output_frame:
        reflect_z_marker_corners_dict_inplace(aligned_markers_in_output_frame)
    apply_z_flip_to_surface_result(surface)


__all__ = [
    "apply_output_z_reflection",
    "apply_z_flip_to_surface_result",
    "reflect_z_marker_corners_dict_inplace",
    "reflect_z_points_inplace",
    "reflect_z_sim3_matrix",
    "reflect_z_T_world_cam_inplace",
]
