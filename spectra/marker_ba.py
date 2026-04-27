"""Joint marker + camera bundle adjustment for stable ArUco 3D positions.

Structural goal
---------------
Guarantee that every detected 2D ArUco corner reprojects to a single,
stable 3D point across all views. Independent DLT triangulation of the 4
corners of a marker (as in :func:`spectra.align.triangulate_markers`) does
*not* enforce this because it neither ties the 4 corners to a rigid square
nor refines the cameras so that their reprojections agree.

This module introduces a single-call joint bundle adjustment that fixes
both issues at once. For each marker we parameterize a 6-DoF world pose
``T_world_marker`` so that the 4 corners are exactly the known square
template of physical edge length ``L`` meters. For each view we
parameterize a 6-DoF SE(3) delta ``Δ_v`` applied on the left:
``T_refined_v = Δ_v · T_backend_v`` (so ``R_ref = R_Δ R_b`` and
``t_ref = R_Δ t_b + t_Δ``). The shared metric scale
``s = m / backend`` is bootstrapped once from the known marker edge length
and the back-end's native-scale DLT triangulation — after the pre-scale,
the BA itself is pure SE(3) in meters.

The BA minimizes the sum of squared reprojection errors (Huber-robust) of
every corner detection across every view, with a weak Gaussian prior on
each ``Δ_v`` to prevent the optimizer from drifting in degenerate /
weakly-observed configurations.

All operations are batched PyTorch ops (GPU when available); the only
per-group work is the initial DLT triangulation (already in
``spectra.align``) and the per-marker Procrustes init below, both of
which run on tiny 4×3 systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from .align import MarkerTriangulation, triangulate_markers
from .aruco import MarkerDetection


# OpenCV ArUco corner ordering (top-left, top-right, bottom-right, bottom-left),
# expressed in the marker's local frame with +X right, +Y up, +Z out-of-plane.
def _marker_template_corners(edge_length_m: float) -> np.ndarray:
    """Return ``[4, 3]`` float64 corner template (meters) for one marker."""
    half = float(edge_length_m) * 0.5
    return np.array(
        [
            [-half, +half, 0.0],  # top-left     (corner 0)
            [+half, +half, 0.0],  # top-right    (corner 1)
            [+half, -half, 0.0],  # bottom-right (corner 2)
            [-half, -half, 0.0],  # bottom-left  (corner 3)
        ],
        dtype=np.float64,
    )


@dataclass
class BundleAdjustmentResult:
    """Output of :func:`joint_bundle_adjust`, all in metric world frame."""

    marker_corners_m: Dict[int, np.ndarray]       # marker_id -> [4, 3] float32
    marker_T_world: Dict[int, np.ndarray]         # marker_id -> [4, 4] float32
    T_world_cam_refined: np.ndarray               # [V, 4, 4] float32
    delta_T_per_view: np.ndarray                  # [V, 4, 4] float32, SE(3) per view
    scale_m_per_backend: float                    # multiply backend world by this to get meters
    initial_reproj_rmse_px: float
    final_reproj_rmse_px: float
    per_marker_reproj_rmse_px: Dict[int, float]
    per_view_reproj_rmse_px: Dict[int, float]
    per_obs_reproj_err_px: np.ndarray             # [M_obs] final per-obs pixel error
    num_observations: int
    num_iters: int
    converged: bool
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level rotation / SE(3) helpers (batched, differentiable).
# ---------------------------------------------------------------------------


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from vectors. ``v``: ``[..., 3]`` -> ``[..., 3, 3]``."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    # fmt: off
    return torch.stack(
        [
            torch.stack([zero,   -z,    y], dim=-1),
            torch.stack([   z, zero,   -x], dim=-1),
            torch.stack([  -y,    x, zero], dim=-1),
        ],
        dim=-2,
    )
    # fmt: on


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched. ``rotvec``: ``[..., 3]`` -> ``[..., 3, 3]``.

    Numerically stable near ``theta=0`` via Taylor expansion of
    ``sin(theta)/theta`` and ``(1-cos(theta))/theta^2``.
    """
    theta2 = (rotvec * rotvec).sum(-1, keepdim=True).clamp_min(1e-30)           # [..., 1]
    theta = torch.sqrt(theta2)                                                  # [..., 1]
    small = theta2 < 1e-8

    # sin(theta)/theta  and (1 - cos(theta))/theta^2
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0,
        torch.sin(theta) / theta,
    )  # [..., 1]
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )  # [..., 1]

    K = _skew(rotvec)                                                           # [..., 3, 3]
    KK = K @ K                                                                  # [..., 3, 3]
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + A[..., None] * K + B[..., None] * KK                           # [..., 3, 3]


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> R^3, batched. ``R``: ``[..., 3, 3]`` -> ``[..., 3]``.

    Delegates to :class:`scipy.spatial.transform.Rotation` for a numerically
    stable implementation across the full 0..pi range.
    """
    from scipy.spatial.transform import Rotation

    R = np.asarray(R, dtype=np.float64)
    leading_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    rotvec_flat = Rotation.from_matrix(R_flat).as_rotvec()                      # [N, 3]
    return rotvec_flat.reshape(*leading_shape, 3).astype(np.float32)


def _so3_mat_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matmul that works for ``[..., 3, 3] @ [..., 3, 3]``."""
    return torch.matmul(A, B)


def _apply_se3_to_points(R: torch.Tensor, t: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """``R @ P + t`` for ``R [..., 3, 3]``, ``t [..., 3]``, ``P [..., 3]``."""
    return torch.einsum("...ij,...j->...i", R, P) + t


def _invert_rigid_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    out = np.zeros_like(T)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3:4] = -out[..., :3, :3] @ t
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Initialization helpers (numpy, CPU, small Nx3 systems).
# ---------------------------------------------------------------------------


def _rigid_fit_procrustes(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``dst ≈ R @ src + t`` (no scale). ``src``, ``dst``: ``[N, 3]``.

    Returns ``(R, t)`` with shapes ``[3, 3]`` and ``[3]``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    H = sc.T @ dc
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R.astype(np.float64), t.astype(np.float64)


def _marker_T_from_triangulated(
    corners_3d_m: np.ndarray,       # [4, 3] float64, in METERS
    edge_length_m: float,
) -> np.ndarray:
    """Build an initial ``T_world_marker`` ``[4,4]`` from 4 (noisy) triangulated corners.

    Procrustes-fits the known rigid-square template onto the triangulated
    corners, yielding a rigid 6-DoF pose even when the 4 points are not
    exactly a square.
    """
    tpl = _marker_template_corners(edge_length_m)
    R, t = _rigid_fit_procrustes(tpl, corners_3d_m)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Observation gathering.
# ---------------------------------------------------------------------------


def _gather_observations(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    marker_ids_kept: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``(view, marker, corner)`` detections into parallel index arrays.

    Returns:
        uv_obs:     ``[M, 2]`` float32, pixel coordinates in the same frame as ``K``.
        view_idx:   ``[M]``   int64.
        marker_idx: ``[M]``   int64, compacted to ``range(len(marker_ids_kept))``.
        corner_idx: ``[M]``   int64, in ``{0, 1, 2, 3}``.
    """
    kept = {int(mid): i for i, mid in enumerate(marker_ids_kept)}
    uv_list: List[np.ndarray] = []
    vidx_list: List[int] = []
    midx_list: List[int] = []
    cidx_list: List[int] = []
    for v, dets in enumerate(detections_per_view):
        for d in dets:
            mid = int(d.id)
            if mid not in kept:
                continue
            uv_list.append(d.corners_xy.astype(np.float32).reshape(4, 2))   # [4, 2]
            vidx_list.extend([v, v, v, v])
            midx_list.extend([kept[mid]] * 4)
            cidx_list.extend([0, 1, 2, 3])

    if not uv_list:
        empty_f = np.zeros((0, 2), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        return empty_f, empty_i, empty_i, empty_i

    uv_obs = np.concatenate(uv_list, axis=0).astype(np.float32)             # [M, 2]
    return (
        uv_obs,
        np.asarray(vidx_list, dtype=np.int64),
        np.asarray(midx_list, dtype=np.int64),
        np.asarray(cidx_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Core bundle adjustment.
# ---------------------------------------------------------------------------


def joint_bundle_adjust(
    detections_per_view: Sequence[Sequence[MarkerDetection]],
    K_per_view: np.ndarray,                         # [V, 3, 3], original image pixels
    T_world_cam_backend: np.ndarray,                # [V, 4, 4], backend native scale
    edge_length_m: float,
    *,
    min_views_per_marker: int = 2,
    max_iters: int = 300,
    lr: float = 5e-3,
    huber_delta_px: float = 2.0,
    cam_prior_sigma_m: float = 0.10,
    cam_prior_sigma_deg: float = 5.0,
    rel_tolerance: float = 1e-7,
    patience: int = 25,
    device: torch.device | str | None = None,
) -> BundleAdjustmentResult:
    """Jointly refine marker poses + camera poses by reprojection minimization.

    Args:
        detections_per_view: list (len V) of per-view ArUco detections.
        K_per_view: [V, 3, 3] intrinsics matching the detection pixel frame.
        T_world_cam_backend: [V, 4, 4] camera-to-world poses from the back-end
            (native scale; the BA bootstraps a metric scale internally).
        edge_length_m: physical ArUco edge length in meters.
        min_views_per_marker: a marker must be seen in >= this many views.
        max_iters, lr: optimizer budget (Adam).
        huber_delta_px: Huber threshold on per-residual pixel error.
        cam_prior_sigma_m / cam_prior_sigma_deg: soft Gaussian prior on the
            translation ``t_Δ`` and rotation vector of ``R_Δ`` in
            ``T_refined = Δ · T_backend`` (keeps the solve anchored to the
            backend when marker coverage is sparse).
        rel_tolerance, patience: early stopping on relative loss change.
        device: optional torch device override; defaults to CUDA if present.

    Returns:
        BundleAdjustmentResult with everything in METRIC world frame.
    """
    V = int(len(detections_per_view))
    if K_per_view.shape != (V, 3, 3):
        raise ValueError(f"Expected K_per_view shape ({V},3,3), got {K_per_view.shape}")
    if T_world_cam_backend.shape != (V, 4, 4):
        raise ValueError(f"Expected T_world_cam_backend shape ({V},4,4), got {T_world_cam_backend.shape}")
    if edge_length_m <= 0:
        raise ValueError(f"edge_length_m must be positive, got {edge_length_m}")

    dev = torch.device(device) if device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    warnings: List[str] = []

    # 1) Triangulate in backend native scale to get an initial (non-rigid) 3D
    #    estimate per marker + a metric scale from median edge length.
    tri = triangulate_markers(
        detections_per_view=detections_per_view,
        K_per_view=K_per_view.astype(np.float64),
        T_world_cam_per_view=T_world_cam_backend.astype(np.float64),
        min_views_per_marker=min_views_per_marker,
    )
    if not tri:
        return _empty_result(V, warnings=["No markers triangulated; skipping BA."])

    marker_ids = sorted(int(mid) for mid in tri.keys())
    M = len(marker_ids)

    # Shared metric scale: pool all finite edges across markers and take the median.
    edges_all = np.concatenate(
        [tri[mid].edge_lengths_3d[np.isfinite(tri[mid].edge_lengths_3d)] for mid in marker_ids],
        axis=0,
    )
    edges_all = edges_all[edges_all > 1e-9]
    if edges_all.size == 0:
        return _empty_result(V, warnings=["Triangulated edges are degenerate; skipping BA."])
    median_edge_backend = float(np.median(edges_all))
    scale_m_per_backend = float(edge_length_m) / median_edge_backend

    # 2) Pre-scale backend cameras to meters.
    T_backend_m = T_world_cam_backend.astype(np.float64).copy()
    T_backend_m[:, :3, 3] *= scale_m_per_backend

    # 3) Marker init: Procrustes-fit the rigid template onto the (now metric)
    #    triangulated corners of each marker.
    T_world_marker_init = np.zeros((M, 4, 4), dtype=np.float64)
    for mi, mid in enumerate(marker_ids):
        corners_m = tri[mid].corners_3d.astype(np.float64) * scale_m_per_backend     # [4, 3] in meters
        T_world_marker_init[mi] = _marker_T_from_triangulated(corners_m, edge_length_m)

    # 4) Gather detections -> flat obs arrays.
    uv_obs_np, view_idx_np, marker_idx_np, corner_idx_np = _gather_observations(
        detections_per_view, marker_ids
    )
    N_obs = int(uv_obs_np.shape[0])
    if N_obs < 6 * M:
        # Not enough observations for a meaningful joint solve — still runs,
        # but the prior will dominate. Leave a warning.
        warnings.append(
            f"Only {N_obs} marker-corner observations for {M} markers — priors will dominate BA."
        )

    # ---- Torch tensors (GPU) -----------------------------------------------
    uv_obs = torch.from_numpy(uv_obs_np).to(dev, dtype=torch.float32)                 # [M_obs, 2]
    view_idx = torch.from_numpy(view_idx_np).to(dev)                                  # [M_obs]
    marker_idx = torch.from_numpy(marker_idx_np).to(dev)                              # [M_obs]
    corner_idx = torch.from_numpy(corner_idx_np).to(dev)                              # [M_obs]
    K = torch.from_numpy(K_per_view.astype(np.float32)).to(dev)                       # [V, 3, 3]

    tpl_np = _marker_template_corners(edge_length_m).astype(np.float32)               # [4, 3]
    template = torch.from_numpy(tpl_np).to(dev)                                       # [4, 3]

    T_backend_m_t = torch.from_numpy(T_backend_m.astype(np.float32)).to(dev)          # [V, 4, 4]
    R_backend = T_backend_m_t[:, :3, :3].contiguous()                                 # [V, 3, 3]
    t_backend = T_backend_m_t[:, :3, 3].contiguous()                                  # [V, 3]

    # ---- Parameters --------------------------------------------------------
    # Per-view SE(3) delta as (rotvec[3], tvec[3]) in the WORLD frame acting on the
    # camera-to-world pose:  T_refined_v = Δ_v · T_backend_v,  with Δ_v ∈ SE(3).
    cam_delta_rot = torch.zeros((V, 3), device=dev, dtype=torch.float32, requires_grad=True)
    cam_delta_t = torch.zeros((V, 3), device=dev, dtype=torch.float32, requires_grad=True)

    # Per-marker 6-DoF world pose, initialized from Procrustes fits.
    marker_rot_init_np = _matrix_to_axis_angle(T_world_marker_init[:, :3, :3])         # [M, 3]
    marker_t_init_np = T_world_marker_init[:, :3, 3].astype(np.float32)                # [M, 3]
    marker_rot = torch.from_numpy(marker_rot_init_np.astype(np.float32)).to(dev).detach().clone().requires_grad_(True)
    marker_t = torch.from_numpy(marker_t_init_np).to(dev).detach().clone().requires_grad_(True)

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        [cam_delta_rot, cam_delta_t, marker_rot, marker_t],
        lr=lr,
    )

    # Prior weights as inverse-variance (consistent with a Gaussian NLL).
    sigma_t = float(cam_prior_sigma_m)
    sigma_r = float(np.deg2rad(cam_prior_sigma_deg))
    inv_var_t = 1.0 / (sigma_t * sigma_t)
    inv_var_r = 1.0 / (sigma_r * sigma_r)

    def _forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, data_loss, per-obs pixel error) for the current iterate."""
        dR_v = _axis_angle_to_matrix(cam_delta_rot)                       # [V, 3, 3]
        # Refined camera-to-world: left-multiply the backend pose by Δ_v ∈ SE(3):
        #   T_refined = Δ_v · T_backend  ⇒  R_new = R_Δ R_b,
        #   t_new = R_Δ t_b + t_Δ
        # so that ``delta_T = T_refined · T_backend^{-1}`` equals Δ_v and point
        # propagation ``p' = Δ_v p`` matches rigid-body kinematics.
        R_new = _so3_mat_mul(dR_v, R_backend)                             # [V, 3, 3]
        t_new = torch.einsum("vij,vj->vi", dR_v, t_backend) + cam_delta_t  # [V, 3]

        # World-to-camera (inverse SE(3)).
        Rwc = R_new.transpose(-1, -2)                                     # [V, 3, 3]
        twc = -torch.einsum("vij,vj->vi", Rwc, t_new)                     # [V, 3]

        Rm = _axis_angle_to_matrix(marker_rot)                            # [M, 3, 3]
        corners_world = torch.einsum("mij,cj->mci", Rm, template) + marker_t[:, None, :]  # [M, 4, 3]

        P_world = corners_world[marker_idx, corner_idx, :]                # [M_obs, 3]

        R_cam_obs = Rwc[view_idx]                                         # [M_obs, 3, 3]
        t_cam_obs = twc[view_idx]                                         # [M_obs, 3]
        P_cam = torch.einsum("oij,oj->oi", R_cam_obs, P_world) + t_cam_obs     # [M_obs, 3]

        z = P_cam[:, 2]
        # Clamp Z to a small positive value so projection + gradients stay finite
        # for any point that happens to land behind the camera during early iters.
        # The hinge term below pushes those points back in front of the camera.
        z_min = 1e-3
        z_safe = z.clamp_min(z_min)
        hinge = torch.clamp(z_min - z, min=0.0)                           # [M_obs], 0 when z >= z_min

        K_obs = K[view_idx]                                               # [M_obs, 3, 3]
        fx = K_obs[:, 0, 0]
        fy = K_obs[:, 1, 1]
        cx = K_obs[:, 0, 2]
        cy = K_obs[:, 1, 2]
        u = fx * (P_cam[:, 0] / z_safe) + cx                              # [M_obs]
        v = fy * (P_cam[:, 1] / z_safe) + cy                              # [M_obs]
        uv_pred = torch.stack([u, v], dim=-1)                             # [M_obs, 2]

        res = uv_pred - uv_obs                                            # [M_obs, 2]
        err_per_obs = torch.linalg.norm(res, dim=-1)                      # [M_obs]

        # Huber on the scalar reprojection error magnitude.
        delta = float(huber_delta_px)
        abs_r = err_per_obs.clamp_min(0.0)
        quad = torch.minimum(abs_r, torch.full_like(abs_r, delta))
        lin = abs_r - quad
        rho = 0.5 * (quad * quad) + delta * lin                           # [M_obs]

        data_loss = rho.sum() + 1.0e4 * (hinge * hinge).sum()

        # Weak prior: shrink deltas toward zero in metric units.
        prior_loss = inv_var_r * (cam_delta_rot * cam_delta_rot).sum() \
                   + inv_var_t * (cam_delta_t * cam_delta_t).sum()
        prior_loss = 0.5 * prior_loss

        total = data_loss + prior_loss
        return total, data_loss, err_per_obs.detach()

    # ---- Run the solver ----------------------------------------------------
    with torch.no_grad():
        _, _, err_initial = _forward()
    initial_rmse = float(torch.sqrt((err_initial ** 2).mean()).item()) if err_initial.numel() > 0 else 0.0

    last_loss = None
    best_loss = float("inf")
    stall = 0
    converged = False
    num_iters_run = 0
    for it in range(int(max_iters)):
        optimizer.zero_grad(set_to_none=True)
        loss, data_loss, _ = _forward()
        if not torch.isfinite(loss):
            warnings.append(f"Non-finite loss at iter {it}; aborting BA.")
            break
        loss.backward()
        optimizer.step()
        num_iters_run = it + 1

        loss_val = float(loss.item())
        if last_loss is not None:
            rel = abs(last_loss - loss_val) / max(abs(last_loss), 1e-12)
            if rel < rel_tolerance:
                stall += 1
                if stall >= int(patience):
                    converged = True
                    break
            else:
                stall = 0
        last_loss = loss_val
        best_loss = min(best_loss, loss_val)

    # ---- Pull refined parameters back to numpy ------------------------------
    # NOTE: the composition MUST match _forward() exactly, otherwise the delta
    # extracted below does not reproduce the refined cameras used for the BA cost.
    with torch.no_grad():
        _, _, err_final = _forward()
        dR_v = _axis_angle_to_matrix(cam_delta_rot)
        R_new = _so3_mat_mul(dR_v, R_backend)
        t_new = torch.einsum("vij,vj->vi", dR_v, t_backend) + cam_delta_t
        Rm = _axis_angle_to_matrix(marker_rot)
        corners_world = torch.einsum("mij,cj->mci", Rm, template) + marker_t[:, None, :]

    R_new_np = R_new.detach().cpu().numpy().astype(np.float32)                            # [V, 3, 3]
    t_new_np = t_new.detach().cpu().numpy().astype(np.float32)                            # [V, 3]
    Rm_np = Rm.detach().cpu().numpy().astype(np.float32)                                  # [M, 3, 3]
    tm_np = marker_t.detach().cpu().numpy().astype(np.float32)                            # [M, 3]
    corners_np = corners_world.detach().cpu().numpy().astype(np.float32)                  # [M, 4, 3]
    err_final_np = err_final.cpu().numpy().astype(np.float32)                             # [M_obs]

    T_world_cam_refined = np.repeat(np.eye(4, dtype=np.float32)[None], V, axis=0)
    T_world_cam_refined[:, :3, :3] = R_new_np
    T_world_cam_refined[:, :3, 3] = t_new_np

    # Δ_v = T_refined_v · T_backend_m_v^{-1}
    T_backend_m_inv = _invert_rigid_4x4(T_backend_m)                                      # [V, 4, 4]
    delta_T = (T_world_cam_refined.astype(np.float64) @ T_backend_m_inv).astype(np.float32)

    marker_corners_m = {int(mid): corners_np[i].copy() for i, mid in enumerate(marker_ids)}
    marker_T_world: Dict[int, np.ndarray] = {}
    for i, mid in enumerate(marker_ids):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rm_np[i]
        T[:3, 3] = tm_np[i]
        marker_T_world[int(mid)] = T

    per_marker_rmse: Dict[int, float] = {}
    per_view_rmse: Dict[int, float] = {}
    if err_final_np.size > 0:
        for i, mid in enumerate(marker_ids):
            mask = marker_idx_np == i
            if np.any(mask):
                per_marker_rmse[int(mid)] = float(np.sqrt(np.mean(err_final_np[mask] ** 2)))
        for v in range(V):
            mask = view_idx_np == v
            if np.any(mask):
                per_view_rmse[int(v)] = float(np.sqrt(np.mean(err_final_np[mask] ** 2)))

    final_rmse = float(np.sqrt(np.mean(err_final_np ** 2))) if err_final_np.size > 0 else 0.0

    return BundleAdjustmentResult(
        marker_corners_m=marker_corners_m,
        marker_T_world=marker_T_world,
        T_world_cam_refined=T_world_cam_refined.astype(np.float32),
        delta_T_per_view=delta_T.astype(np.float32),
        scale_m_per_backend=float(scale_m_per_backend),
        initial_reproj_rmse_px=initial_rmse,
        final_reproj_rmse_px=final_rmse,
        per_marker_reproj_rmse_px=per_marker_rmse,
        per_view_reproj_rmse_px=per_view_rmse,
        per_obs_reproj_err_px=err_final_np,
        num_observations=N_obs,
        num_iters=num_iters_run,
        converged=converged,
        warnings=warnings,
    )


def _empty_result(V: int, warnings: List[str]) -> BundleAdjustmentResult:
    return BundleAdjustmentResult(
        marker_corners_m={},
        marker_T_world={},
        T_world_cam_refined=np.repeat(np.eye(4, dtype=np.float32)[None], V, axis=0),
        delta_T_per_view=np.repeat(np.eye(4, dtype=np.float32)[None], V, axis=0),
        scale_m_per_backend=1.0,
        initial_reproj_rmse_px=float("nan"),
        final_reproj_rmse_px=float("nan"),
        per_marker_reproj_rmse_px={},
        per_view_reproj_rmse_px={},
        per_obs_reproj_err_px=np.zeros((0,), dtype=np.float32),
        num_observations=0,
        num_iters=0,
        converged=False,
        warnings=list(warnings),
    )


# ---------------------------------------------------------------------------
# Numpy helpers to apply the refined SE(3) delta + scale downstream.
# ---------------------------------------------------------------------------


def apply_delta_and_scale_to_points(
    points_backend_world: np.ndarray,   # [..., 3]
    view_idx_of_points: np.ndarray,     # [..., ]  integer array, same leading shape as points[..., 0]
    delta_T_per_view: np.ndarray,       # [V, 4, 4] (already in meters)
    scale_m_per_backend: float,
) -> np.ndarray:
    """Transform per-view backend-world points into the BA-refined metric world.

    For every point ``p_v`` attributed to view ``v``:
        ``p_refined = Δ_v @ (s · p_v)``

    This is the correct composition when backend cameras were pre-scaled by
    ``s`` before BA: then ``Δ_v`` lives in meters and ``s · p_v`` pulls the
    points into meters too.
    """
    pts = np.asarray(points_backend_world, dtype=np.float32)
    idx = np.asarray(view_idx_of_points, dtype=np.int64)
    if pts.shape[:-1] != idx.shape:
        raise ValueError(
            f"Expected matching leading shape between points {pts.shape} and view idx {idx.shape}"
        )
    if pts.size == 0:
        return pts.copy()

    pts_m = pts.astype(np.float64) * float(scale_m_per_backend)
    D = delta_T_per_view.astype(np.float64)                                             # [V, 4, 4]
    # Gather per-point 4x4, then rigid-transform.
    D_per_pt = D[idx]                                                                   # [..., 4, 4]
    R = D_per_pt[..., :3, :3]
    t = D_per_pt[..., :3, 3]
    out = np.einsum("...ij,...j->...i", R, pts_m) + t
    return out.astype(np.float32)


__all__ = [
    "BundleAdjustmentResult",
    "apply_delta_and_scale_to_points",
    "joint_bundle_adjust",
]
