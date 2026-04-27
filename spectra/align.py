"""Multi-view triangulation of ArUco corners, plane fit, and Sim3 alignment.

This module ties the 2D ArUco detections from ``spectra.aruco`` to the 3D
reconstruction produced by the MASt3R back-end. The high-level
pipeline is:

1. `triangulate_markers(...)` — linear DLT triangulation per corner across
   all views that saw the marker, using per-view intrinsics and extrinsics.
2. `estimate_metric_scale(...)` — robust median scale recovery from the
   known physical edge length.
3. `fit_marker_plane(...)` — SVD plane fit across all triangulated corners,
   giving the Z axis of the output frame.
4. `build_sim3_to_aruco_frame(...)` — compose scale + rotation + translation
   so the ArUco plane sits at Z=0 with Z pointing up.
5. `apply_similarity_*` helpers — apply the Sim3 to clouds, per-view point
   maps, and camera poses.

Everything is vectorized with NumPy; no per-point Python loops on the hot
path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, TypedDict

import numpy as np

from .aruco import MarkerDetection

if TYPE_CHECKING:
    from .marker_ba import BundleAdjustmentResult


@dataclass(frozen=True)
class PlaneFrame:
    """Output-frame definition: origin + 3x3 basis (columns = U, V, N)."""

    origin: np.ndarray   # shape: (3,), float32
    basis: np.ndarray    # shape: (3, 3), float32, columns are u, v, n (n = normal)

    @property
    def normal(self) -> np.ndarray:
        return self.basis[:, 2].astype(np.float32)

    @property
    def tangent_u(self) -> np.ndarray:
        return self.basis[:, 0].astype(np.float32)

    @property
    def tangent_v(self) -> np.ndarray:
        return self.basis[:, 1].astype(np.float32)


@dataclass
class MarkerTriangulation:
    """Triangulated 3D corners for one marker ID."""

    marker_id: int
    corners_3d: np.ndarray              # shape: (4, 3), float32
    num_views: int                      # number of views that observed this marker
    reproj_rmse_px: float               # mean reprojection RMSE across observations (pixels)
    edge_lengths_3d: np.ndarray         # shape: (4,), float32, edge lengths in 3D
    center_3d: np.ndarray               # shape: (3,), float32


@dataclass
class ArucoAlignment:
    """Result of the full ArUco -> alignment pipeline."""

    markers: Dict[int, MarkerTriangulation]
    plane_frame: PlaneFrame | None
    scale: float                          # s such that x_metric = s * x_input
    scale_mad: float                      # robust dispersion of per-marker scales
    sim3: np.ndarray                      # shape: (4, 4), float32, x' = s * R * x + t
    sim3_scale: float                     # convenience: just the scale
    rotation: np.ndarray                  # shape: (3, 3), float32
    translation: np.ndarray               # shape: (3,), float32
    used_marker_ids: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Populated when `align_with_aruco(..., use_bundle_adjustment=True)`. When
    # present, the pipeline should apply ``bundle_adjustment.delta_T_per_view``
    # + ``bundle_adjustment.scale_m_per_backend`` to the back-end's per-view
    # point maps before fusion so the whole reconstruction stays consistent
    # with the stable ArUco corners.
    bundle_adjustment: Any = None  # Optional[BundleAdjustmentResult] — stringified to avoid cycle


def _invert_rigid_4x4(T: np.ndarray) -> np.ndarray:
    """Invert a [4,4] rigid transform (R,t). Batched along leading axes."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [...,4,4] transform, got {T.shape}")
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = np.swapaxes(R, -1, -2)  # [..., 3, 3]
    t_inv = -R_inv @ t              # [..., 3, 1]
    out = np.zeros_like(T)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:4] = t_inv
    out[..., 3, 3] = 1.0
    return out


def _build_projection_matrices(
    K_per_view: np.ndarray,
    T_world_cam_per_view: np.ndarray,
) -> np.ndarray:
    """Compose projection matrices `P = K @ [R | t]` that map world -> pixel.

    Args:
        K_per_view: shape [V, 3, 3], intrinsics in the image coord system
            used by the 2D detections.
        T_world_cam_per_view: shape [V, 4, 4], camera-to-world poses.

    Returns:
        P: shape [V, 3, 4], projection matrices mapping homogeneous world
        points to homogeneous pixel coordinates.
    """
    T_cam_world = _invert_rigid_4x4(T_world_cam_per_view)  # [V,4,4]
    Rt = T_cam_world[..., :3, :]                           # [V,3,4]
    P = np.einsum("vij,vjk->vik", K_per_view.astype(np.float64), Rt)
    return P


def _dlt_triangulate_batch(
    uv: np.ndarray,       # [M, 2] pixel observations
    proj: np.ndarray,     # [M, 3, 4] per-observation projection matrices
    groups: np.ndarray,   # [M] int64, group ID for each obs (one group per 3D point)
    num_groups: int,
) -> np.ndarray:
    """Solve one DLT per group by stacking `A @ X = 0` and taking SVD.

    Each observation contributes 2 equations:
        u * P_row2 - P_row0 = 0
        v * P_row2 - P_row1 = 0
    For each group we pick its rows (2 per obs) and solve for X in homogeneous
    coords via the right-singular-vector of the smallest singular value.

    This loops over groups because the number of observations per group
    varies, but builds the equations vectorized inside each group.
    """
    p_row0 = proj[:, 0, :]  # [M, 4]
    p_row1 = proj[:, 1, :]  # [M, 4]
    p_row2 = proj[:, 2, :]  # [M, 4]
    u = uv[:, 0:1]          # [M, 1]
    v = uv[:, 1:2]          # [M, 1]

    # Two equations per observation.
    eq1 = u * p_row2 - p_row0  # [M, 4]
    eq2 = v * p_row2 - p_row1  # [M, 4]
    eqs = np.stack([eq1, eq2], axis=1).reshape(-1, 4)  # [2M, 4]
    group_per_row = np.repeat(groups, 2)                # [2M]

    points_3d = np.zeros((num_groups, 3), dtype=np.float64)
    for g in range(num_groups):
        A = eqs[group_per_row == g]
        if A.shape[0] < 4:
            points_3d[g] = np.nan
            continue
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        X = Vh[-1]
        if abs(X[3]) < 1e-12:
            points_3d[g] = np.nan
            continue
        points_3d[g] = X[:3] / X[3]
    return points_3d


def _reprojection_error_per_obs(
    points_3d: np.ndarray,   # [G, 3]
    uv: np.ndarray,          # [M, 2]
    proj: np.ndarray,        # [M, 3, 4]
    groups: np.ndarray,      # [M]
) -> np.ndarray:
    """Return per-observation reprojection error (pixels), shape [M]."""
    X = points_3d[groups]                        # [M, 3]
    X_h = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # [M, 4]
    uvw = np.einsum("mij,mj->mi", proj, X_h)     # [M, 3]
    w = uvw[:, 2:3]
    valid = np.abs(w.squeeze(-1)) > 1e-8
    uv_hat = np.full_like(uv, np.nan)
    uv_hat[valid] = uvw[valid, :2] / w[valid]
    err = np.linalg.norm(uv_hat - uv, axis=1)    # [M]
    return err


def triangulate_markers(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    K_per_view: np.ndarray,                     # [V, 3, 3]
    T_world_cam_per_view: np.ndarray,           # [V, 4, 4]
    min_views_per_marker: int = 2,
) -> Dict[int, MarkerTriangulation]:
    """Multi-view triangulate every marker corner seen in enough views.

    Each marker has 4 corners, and each is triangulated independently via
    linear DLT. The caller must ensure that `K_per_view` is expressed in
    the same image coordinate system used for the 2D corner coordinates
    inside `detections_per_view`.
    """
    num_views = len(detections_per_view)
    if K_per_view.shape != (num_views, 3, 3):
        raise ValueError(f"Expected K_per_view shape ({num_views},3,3), got {K_per_view.shape}")
    if T_world_cam_per_view.shape != (num_views, 4, 4):
        raise ValueError(
            f"Expected T_world_cam_per_view shape ({num_views},4,4), got {T_world_cam_per_view.shape}"
        )

    P_per_view = _build_projection_matrices(K_per_view, T_world_cam_per_view)  # [V,3,4]

    # Group observations by (marker_id, corner_idx).
    ids_seen: Dict[int, int] = {}                # marker_id -> views-count
    obs_rows: List[dict] = []
    for view_idx, dets in enumerate(detections_per_view):
        for detection in dets:
            ids_seen[detection.id] = ids_seen.get(detection.id, 0) + 1
            for corner_idx in range(4):
                obs_rows.append(
                    {
                        "marker_id": int(detection.id),
                        "corner_idx": int(corner_idx),
                        "view_idx": int(view_idx),
                        "uv": detection.corners_xy[corner_idx].astype(np.float64),
                    }
                )

    if not obs_rows:
        return {}

    # Filter markers by min_views_per_marker (applies to all 4 corners uniformly).
    valid_marker_ids = {mid for mid, count in ids_seen.items() if count >= max(min_views_per_marker, 2)}
    obs_rows = [row for row in obs_rows if row["marker_id"] in valid_marker_ids]
    if not obs_rows:
        return {}

    marker_ids_sorted = sorted(valid_marker_ids)
    marker_to_index = {mid: i for i, mid in enumerate(marker_ids_sorted)}

    num_markers = len(marker_ids_sorted)
    num_groups = num_markers * 4  # marker_idx * 4 + corner_idx

    uv = np.stack([row["uv"] for row in obs_rows], axis=0)            # [M,2]
    view_idx = np.array([row["view_idx"] for row in obs_rows], dtype=np.int64)  # [M]
    group_ids = np.array(
        [marker_to_index[row["marker_id"]] * 4 + row["corner_idx"] for row in obs_rows],
        dtype=np.int64,
    )
    proj = P_per_view[view_idx]                                       # [M,3,4]

    points_3d = _dlt_triangulate_batch(uv, proj, group_ids, num_groups=num_groups)  # [G,3]
    errs = _reprojection_error_per_obs(points_3d, uv, proj, group_ids)              # [M]

    results: Dict[int, MarkerTriangulation] = {}
    for marker_id in marker_ids_sorted:
        m_idx = marker_to_index[marker_id]
        corners_3d = points_3d[m_idx * 4 : m_idx * 4 + 4].astype(np.float32)  # [4,3]
        if not np.isfinite(corners_3d).all():
            continue

        mask = np.isin(group_ids, np.arange(m_idx * 4, m_idx * 4 + 4))
        marker_errs = errs[mask]
        rmse = float(np.sqrt(np.nanmean(marker_errs ** 2))) if marker_errs.size > 0 else float("nan")

        # Edge lengths: 0-1, 1-2, 2-3, 3-0
        edges = np.linalg.norm(corners_3d[[1, 2, 3, 0]] - corners_3d[[0, 1, 2, 3]], axis=1).astype(np.float32)
        center = corners_3d.mean(axis=0).astype(np.float32)

        results[marker_id] = MarkerTriangulation(
            marker_id=int(marker_id),
            corners_3d=corners_3d,
            num_views=int(ids_seen[marker_id]),
            reproj_rmse_px=rmse,
            edge_lengths_3d=edges,
            center_3d=center,
        )
    return results


def estimate_metric_scale(
    markers: Mapping[int, MarkerTriangulation],
    edge_length_m: float,
) -> tuple[float, float]:
    """Robust metric scale: returns ``(scale, mad)``.

    `scale` is such that ``x_metric = scale * x_input``.
    """
    if edge_length_m <= 0:
        raise ValueError(f"edge_length_m must be positive, got {edge_length_m}")
    if not markers:
        return 1.0, float("inf")

    # Use per-marker median edge to be robust to degenerate triangulations.
    per_marker_scales = []
    for m in markers.values():
        edges = m.edge_lengths_3d[np.isfinite(m.edge_lengths_3d)]
        edges = edges[edges > 1e-9]
        if edges.size == 0:
            continue
        median_edge = float(np.median(edges))
        if median_edge <= 0:
            continue
        per_marker_scales.append(float(edge_length_m) / median_edge)

    if not per_marker_scales:
        return 1.0, float("inf")

    scales = np.asarray(per_marker_scales, dtype=np.float64)
    med = float(np.median(scales))
    mad = float(np.median(np.abs(scales - med)))
    return med, mad


def fit_marker_plane(
    markers: Mapping[int, MarkerTriangulation],
    origin_marker_id: int | None = None,
    reference_up: np.ndarray | None = None,
) -> PlaneFrame:
    """SVD plane fit across all ArUco corners, returned as a PlaneFrame.

    - The plane normal becomes the Z axis of the output frame.
    - The sign is chosen so the normal agrees with ``reference_up`` (default
      +Z of the input frame); this prevents flipping between runs.
    - If `origin_marker_id` is given, the frame origin is the centroid of
      that marker's 4 corners. Otherwise it's the centroid of every corner.
    - The in-plane orientation is recovered from either the chosen origin
      marker's top edge, or the dominant PCA tangent of the pooled corners.
    """
    if not markers:
        raise ValueError("Cannot fit a plane from an empty set of markers")

    all_corners = np.concatenate([m.corners_3d for m in markers.values()], axis=0).astype(np.float64)  # [4*M,3]
    if all_corners.shape[0] < 3:
        raise ValueError("Need at least 3 corner points to fit a plane")

    centroid = all_corners.mean(axis=0)  # [3]

    if origin_marker_id is not None and origin_marker_id in markers:
        origin = markers[origin_marker_id].center_3d.astype(np.float64)
        orientation_marker = markers[origin_marker_id]
    else:
        origin = centroid
        orientation_marker = None

    centered = all_corners - centroid
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    normal = Vh[-1]                          # smallest-variance axis
    normal = normal / max(np.linalg.norm(normal), 1e-12)

    ref_up = np.array([0.0, 0.0, 1.0]) if reference_up is None else np.asarray(reference_up, dtype=np.float64)
    ref_up = ref_up / max(np.linalg.norm(ref_up), 1e-12)
    if float(normal @ ref_up) < 0.0:
        normal = -normal

    # In-plane tangent: prefer the origin marker's top edge (corner 0 -> corner 1).
    tangent_u = None
    if orientation_marker is not None:
        edge_vec = (
            orientation_marker.corners_3d[1].astype(np.float64)
            - orientation_marker.corners_3d[0].astype(np.float64)
        )
        edge_in_plane = edge_vec - normal * float(edge_vec @ normal)
        if np.linalg.norm(edge_in_plane) > 1e-6:
            tangent_u = edge_in_plane / np.linalg.norm(edge_in_plane)

    if tangent_u is None:
        # PCA in the plane: project centered corners, take the top singular
        # direction, and keep the component orthogonal to `normal`.
        proj = centered - np.outer(centered @ normal, normal)  # [N,3]
        _, _, Vh_plane = np.linalg.svd(proj, full_matrices=False)
        tangent_u = Vh_plane[0]
        tangent_u = tangent_u - normal * float(tangent_u @ normal)
        tangent_u = tangent_u / max(np.linalg.norm(tangent_u), 1e-12)

    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(np.linalg.norm(tangent_v), 1e-12)
    tangent_u = np.cross(tangent_v, normal)
    tangent_u = tangent_u / max(np.linalg.norm(tangent_u), 1e-12)

    basis = np.stack([tangent_u, tangent_v, normal], axis=1).astype(np.float32)
    return PlaneFrame(origin=origin.astype(np.float32), basis=basis)


def build_sim3_to_aruco_frame(
    plane_frame: PlaneFrame,
    scale: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Build the Sim3 that maps the reconstruction frame to the ArUco frame.

    The output convention is: ``x' = scale * R @ x + t``.

    In the output frame:
        - ``plane_frame.origin`` goes to the origin (0, 0, 0).
        - ``plane_frame.tangent_u`` becomes +X.
        - ``plane_frame.tangent_v`` becomes +Y.
        - ``plane_frame.normal``    becomes +Z.
    """
    basis = np.asarray(plane_frame.basis, dtype=np.float64)  # columns = u,v,n
    # Output = basis^T @ (x - origin); equivalently R = basis^T, t = -R @ origin, then scaled.
    R = basis.T                                              # [3,3]
    origin = np.asarray(plane_frame.origin, dtype=np.float64)
    t = -R @ origin                                          # [3]
    s = float(scale)
    t_scaled = s * t
    return s, R.astype(np.float32), t_scaled.astype(np.float32)


def build_similarity_matrix(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack ``(scale, R, t)`` into a [4,4] matrix (used for logging)."""
    S = np.eye(4, dtype=np.float32)
    S[:3, :3] = float(scale) * np.asarray(R, dtype=np.float32)
    S[:3, 3] = np.asarray(t, dtype=np.float32)
    return S


def apply_similarity_to_points(
    points: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply ``x' = s * R @ x + t`` to points of any shape ``[..., 3]``."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.copy()
    orig_shape = pts.shape
    pts_flat = pts.reshape(-1, 3).astype(np.float64)
    R64 = np.asarray(R, dtype=np.float64)
    t64 = np.asarray(t, dtype=np.float64)
    out = float(scale) * (pts_flat @ R64.T) + t64[None, :]
    return out.astype(np.float32).reshape(orig_shape)


def apply_similarity_to_camera_poses(
    T_world_cam: np.ndarray,   # [V,4,4]
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Transform camera-to-world poses under the Sim3 ``x' = s * R @ x + t``.

    The camera center is mapped with the full Sim3; the camera orientation
    is rotated by R.
    """
    T = np.asarray(T_world_cam, dtype=np.float32)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"Expected [V,4,4] poses, got {T.shape}")
    R64 = np.asarray(R, dtype=np.float64)
    t64 = np.asarray(t, dtype=np.float64)
    s = float(scale)

    out = np.repeat(np.eye(4, dtype=np.float32)[None], T.shape[0], axis=0)  # [V,4,4]
    R_wc = T[:, :3, :3].astype(np.float64)  # [V,3,3]
    t_wc = T[:, :3, 3].astype(np.float64)   # [V,3]

    out[:, :3, :3] = np.einsum("ij,vjk->vik", R64, R_wc).astype(np.float32)
    out[:, :3, 3] = (s * np.einsum("ij,vj->vi", R64, t_wc) + t64[None, :]).astype(np.float32)
    return out


def apply_similarity_to_intrinsics(
    K_per_view: np.ndarray,  # [V,3,3]
) -> np.ndarray:
    """Sim3 does not change intrinsics — just return a copy.

    Kept as a named helper for API symmetry with the other apply-helpers.
    """
    return np.asarray(K_per_view, dtype=np.float32).copy()


def _markers_from_ba(
    ba: "BundleAdjustmentResult",
    edge_length_m: float,
) -> Dict[int, MarkerTriangulation]:
    """Wrap rigid-square BA corners into ``MarkerTriangulation`` objects.

    The corners come from the BA-optimized per-marker 6-DoF pose times the
    metric square template, so every marker has exact ``edge_length_m``
    edges by construction. ``reproj_rmse_px`` is the per-marker final BA
    residual (pixels).
    """
    out: Dict[int, MarkerTriangulation] = {}
    for mid, corners in ba.marker_corners_m.items():
        edges = np.linalg.norm(
            corners[[1, 2, 3, 0]] - corners[[0, 1, 2, 3]], axis=1
        ).astype(np.float32)
        center = corners.mean(axis=0).astype(np.float32)
        out[int(mid)] = MarkerTriangulation(
            marker_id=int(mid),
            corners_3d=corners.astype(np.float32),
            num_views=1,  # BA pools across views; exact per-marker view count is in ba diagnostics
            reproj_rmse_px=float(ba.per_marker_reproj_rmse_px.get(int(mid), float("nan"))),
            edge_lengths_3d=edges,
            center_3d=center,
        )
    return out


def align_with_aruco(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    K_per_view: np.ndarray,
    T_world_cam_per_view: np.ndarray,
    edge_length_m: float,
    origin_marker_id: int | None = None,
    min_views_per_marker: int = 2,
    enforce_metric_scale: bool = True,
    use_bundle_adjustment: bool = False,
    ba_options: Mapping[str, Any] | None = None,
) -> ArucoAlignment:
    """One-call convenience: triangulate (or jointly BA) + fit plane + build Sim3.

    When ``use_bundle_adjustment=True``, runs
    :func:`spectra.marker_ba.joint_bundle_adjust` to jointly refine per-marker
    6-DoF poses (rigid squares of known edge length) and per-view SE(3)
    camera deltas by minimizing reprojection errors of all corner detections.
    The resulting ``ArucoAlignment.markers`` then hold stable rigid-square
    corners in METRIC world, and ``ArucoAlignment.bundle_adjustment`` exposes
    the per-view delta + scale that the pipeline must propagate to the
    back-end's fused cloud so the whole scene stays coherent.

    When ``use_bundle_adjustment=False`` (default, backward compatible),
    behaves exactly as before: DLT triangulation + median-edge scale.

    If ``enforce_metric_scale`` is False, the returned Sim3 uses scale=1 and
    only rotates/translates — useful when the input frame is already metric
    (e.g. right after BA, which always produces metric output).
    """
    warnings: List[str] = []
    ba_result: "BundleAdjustmentResult | None" = None
    scale: float
    mad: float

    if use_bundle_adjustment:
        # Local import to avoid a circular dependency at module load time.
        from .marker_ba import joint_bundle_adjust

        opts = dict(ba_options or {})
        ba_result = joint_bundle_adjust(
            detections_per_view=detections_per_view,
            K_per_view=np.asarray(K_per_view, dtype=np.float32),
            T_world_cam_backend=np.asarray(T_world_cam_per_view, dtype=np.float32),
            edge_length_m=float(edge_length_m),
            min_views_per_marker=min_views_per_marker,
            **opts,
        )
        warnings.extend(ba_result.warnings)

        if not ba_result.marker_corners_m:
            return ArucoAlignment(
                markers={},
                plane_frame=None,
                scale=1.0,
                scale_mad=float("inf"),
                sim3=np.eye(4, dtype=np.float32),
                sim3_scale=1.0,
                rotation=np.eye(3, dtype=np.float32),
                translation=np.zeros(3, dtype=np.float32),
                used_marker_ids=[],
                warnings=warnings + ["Bundle adjustment produced no markers."],
                bundle_adjustment=ba_result,
            )

        markers = _markers_from_ba(ba_result, edge_length_m=edge_length_m)

        # BA runs in meters already; the Sim3 to ArUco frame is a pure SE(3).
        # ``scale`` still exposed for API symmetry (input -> meters = scale * x_input);
        # here ``x_input`` is the caller's (backend-scaled-to-meters) frame, i.e. scale=1.
        scale = 1.0
        edges_all = np.concatenate([m.edge_lengths_3d for m in markers.values()], axis=0)
        edges_all = edges_all[np.isfinite(edges_all) & (edges_all > 0)]
        if edges_all.size > 0:
            # Residual edge dispersion (should be numerically zero for rigid squares).
            med = float(np.median(edges_all))
            mad = float(np.median(np.abs(edges_all - med)))
        else:
            mad = float("inf")

        # After BA, enforce_metric_scale is already realized (frame is in meters):
        effective_scale = 1.0
    else:
        markers = triangulate_markers(
            detections_per_view=detections_per_view,
            K_per_view=np.asarray(K_per_view, dtype=np.float64),
            T_world_cam_per_view=np.asarray(T_world_cam_per_view, dtype=np.float64),
            min_views_per_marker=min_views_per_marker,
        )

        if not markers:
            warnings.append("No ArUco markers could be triangulated; skipping alignment.")
            return ArucoAlignment(
                markers={},
                plane_frame=None,
                scale=1.0,
                scale_mad=float("inf"),
                sim3=np.eye(4, dtype=np.float32),
                sim3_scale=1.0,
                rotation=np.eye(3, dtype=np.float32),
                translation=np.zeros(3, dtype=np.float32),
                used_marker_ids=[],
                warnings=warnings,
            )

        scale, mad = estimate_metric_scale(markers, edge_length_m=edge_length_m)
        if not math.isfinite(scale) or scale <= 0:
            warnings.append(f"Scale estimation failed ({scale}); falling back to scale=1.0.")
            scale = 1.0

        effective_scale = float(scale) if enforce_metric_scale else 1.0

    plane_frame = fit_marker_plane(markers, origin_marker_id=origin_marker_id)

    s, R, t = build_sim3_to_aruco_frame(plane_frame, scale=effective_scale)
    sim3 = build_similarity_matrix(s, R, t)

    return ArucoAlignment(
        markers=markers,
        plane_frame=plane_frame,
        scale=float(scale),
        scale_mad=float(mad),
        sim3=sim3,
        sim3_scale=float(effective_scale),
        rotation=R,
        translation=t,
        used_marker_ids=sorted(markers.keys()),
        warnings=warnings,
        bundle_adjustment=ba_result,
    )


class MarkerReprojectionStats(TypedDict):
    """Diagnostics from :func:`marker_corner_reprojection_stats`."""

    rmse_px: float
    max_px: float
    p95_px: float
    num_observations: int


def markers_best_fit_plane_rms_m(
    marker_corners_world: Mapping[int, np.ndarray],
) -> float:
    """RMS distance (meters) of every corner to the best-fit plane.

    Single-sheet layouts stay below a few millimetres; multi-planar boards
    (different walls) register much larger values — not an algorithm bug.
    """
    if not marker_corners_world:
        return float("nan")
    all_c = np.concatenate(
        [np.asarray(v, dtype=np.float64).reshape(4, 3) for v in marker_corners_world.values()],
        axis=0,
    )  # [N, 3]
    if all_c.shape[0] < 3:
        return float("nan")
    c = all_c - all_c.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    n = vh[-1]
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    dist = np.abs(c @ n)
    return float(np.sqrt(np.mean(dist * dist)))


def per_view_marker_corner_rmse_px(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    marker_corners_world: Mapping[int, np.ndarray],
    K_per_view: np.ndarray,
    T_world_cam: np.ndarray,
) -> np.ndarray:
    """Per-view RMSE (pixels) over all marker corners; NaN if no markers in that view.

    Shape ``[V]`` float32. Uses the same projection model as bundle adjustment /
    :func:`marker_corner_reprojection_stats`.
    """
    V = len(detections_per_view)
    if K_per_view.shape[0] != V or T_world_cam.shape[0] != V:
        raise ValueError("K_per_view / T_world_cam must have one row per view.")

    sum_sq = np.zeros(V, dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    T_cw = _invert_rigid_4x4(np.asarray(T_world_cam, dtype=np.float64))

    for v, dets in enumerate(detections_per_view):
        R = T_cw[v, :3, :3]
        tvec = T_cw[v, :3, 3]
        K = np.asarray(K_per_view[v], dtype=np.float64)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        for d in dets:
            mid = int(d.id)
            if mid not in marker_corners_world:
                continue
            pw = np.asarray(marker_corners_world[mid], dtype=np.float64).reshape(4, 3)
            pc = pw @ R.T + tvec
            z = pc[:, 2]
            if np.any(z <= 1e-8):
                continue
            u = fx * pc[:, 0] / z + cx
            vpix = fy * pc[:, 1] / z + cy
            uv = np.stack([u, vpix], axis=1)
            gt = np.asarray(d.corners_xy, dtype=np.float64).reshape(4, 2)
            err = np.linalg.norm(uv - gt, axis=1)
            sum_sq[v] += float(np.sum(err * err))
            counts[v] += 4

    out = np.full(V, np.nan, dtype=np.float64)
    m = counts > 0
    out[m] = np.sqrt(sum_sq[m] / counts[m])
    return out.astype(np.float32)


def marker_corner_reprojection_stats(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    marker_corners_world: Mapping[int, np.ndarray],
    K_per_view: np.ndarray,
    T_world_cam: np.ndarray,
) -> MarkerReprojectionStats:
    """Mean / max pixel error projecting ``marker_corners_world`` with ``T,K``.

    Use this **after** Sim3 so ``marker_corners_world`` matches ``cloud.ply`` and
    ``camera_poses_output_frame.npy`` — the same frame Rerun uses for ``/aruco``.
    """
    V = len(detections_per_view)
    if K_per_view.shape[0] != V or T_world_cam.shape[0] != V:
        raise ValueError("K_per_view / T_world_cam must have one row per view.")

    T_cw = _invert_rigid_4x4(np.asarray(T_world_cam, dtype=np.float64))  # [V, 4, 4]
    errs: List[float] = []

    for v, dets in enumerate(detections_per_view):
        R = T_cw[v, :3, :3]
        tvec = T_cw[v, :3, 3]
        K = np.asarray(K_per_view[v], dtype=np.float64)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        for d in dets:
            mid = int(d.id)
            if mid not in marker_corners_world:
                continue
            pw = np.asarray(marker_corners_world[mid], dtype=np.float64).reshape(4, 3)
            pc = pw @ R.T + tvec
            z = pc[:, 2]
            if np.any(z <= 1e-8):
                continue
            u = fx * pc[:, 0] / z + cx
            vpix = fy * pc[:, 1] / z + cy
            uv = np.stack([u, vpix], axis=1)
            gt = np.asarray(d.corners_xy, dtype=np.float64).reshape(4, 2)
            errs.extend(np.linalg.norm(uv - gt, axis=1).tolist())

    if not errs:
        return MarkerReprojectionStats(
            rmse_px=float("nan"),
            max_px=float("nan"),
            p95_px=float("nan"),
            num_observations=0,
        )
    e = np.asarray(errs, dtype=np.float64)
    return MarkerReprojectionStats(
        rmse_px=float(np.sqrt(np.mean(e * e))),
        max_px=float(np.max(e)),
        p95_px=float(np.percentile(e, 95.0)),
        num_observations=int(e.shape[0]),
    )


__all__ = [
    "ArucoAlignment",
    "MarkerTriangulation",
    "PlaneFrame",
    "align_with_aruco",
    "apply_similarity_to_camera_poses",
    "apply_similarity_to_intrinsics",
    "apply_similarity_to_points",
    "build_similarity_matrix",
    "build_sim3_to_aruco_frame",
    "estimate_metric_scale",
    "fit_marker_plane",
    "marker_corner_reprojection_stats",
    "markers_best_fit_plane_rms_m",
    "per_view_marker_corner_rmse_px",
    "triangulate_markers",
    "MarkerReprojectionStats",
]
