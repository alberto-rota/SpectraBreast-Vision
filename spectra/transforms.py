"""Pose/transform utilities shared across pipeline modules."""

from __future__ import annotations

import math

import torch


def xyzeuler_to_hmat(
    poses: torch.Tensor,
    angles_in_degrees: bool = False,
    convention: str = "XYZ",
    translation_scale: float = 1.0,
) -> torch.Tensor:
    """Convert (..., N, 6) XYZ+Euler poses to (..., N, 4, 4) homogeneous transforms."""
    assert poses.shape[-1] == 6, "Expected last dim = 6 for [x, y, z, a1, a2, a3]"

    device = poses.device
    dtype = poses.dtype
    shape_prefix = poses.shape[:-1]  # (..., N)

    x, y, z, a1, a2, a3 = poses.unbind(dim=-1)  # each: (..., N)

    x = x * translation_scale
    y = y * translation_scale
    z = z * translation_scale

    if angles_in_degrees:
        deg_to_rad = math.pi / 180.0
        a1 = a1 * deg_to_rad
        a2 = a2 * deg_to_rad
        a3 = a3 * deg_to_rad

    conv_upper = convention.upper()
    if conv_upper in ["RPY", "ROLLPITCHYAW"]:
        # Roll(X), Pitch(Y), Yaw(Z): R = Rz @ Ry @ Rx
        R_x = _axis_angle_to_matrix("X", a1)  # (..., N, 3, 3)
        R_y = _axis_angle_to_matrix("Y", a2)  # (..., N, 3, 3)
        R_z = _axis_angle_to_matrix("Z", a3)  # (..., N, 3, 3)
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))  # (..., N, 3, 3)
    else:
        assert len(conv_upper) == 3 and all(c in "XYZ" for c in conv_upper), (
            f"Invalid convention {convention}"
        )
        R1 = _axis_angle_to_matrix(conv_upper[0], a1)  # (..., N, 3, 3)
        R2 = _axis_angle_to_matrix(conv_upper[1], a2)  # (..., N, 3, 3)
        R3 = _axis_angle_to_matrix(conv_upper[2], a3)  # (..., N, 3, 3)
        R = torch.matmul(R1, torch.matmul(R2, R3))  # (..., N, 3, 3)

    T = (
        torch.eye(4, dtype=dtype, device=device)
        .view(*[1] * len(shape_prefix), 4, 4)
        .expand(*shape_prefix, 4, 4)
        .clone()
    )  # (..., N, 4, 4)
    T[..., :3, :3] = R
    T[..., :3, 3] = torch.stack((x, y, z), dim=-1)  # (..., N, 3)
    return T


def _axis_angle_to_matrix(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """Generate batch rotation matrices around X/Y/Z for theta (..., N)."""
    cos_t = torch.cos(theta)  # (..., N)
    sin_t = torch.sin(theta)  # (..., N)
    ones = torch.ones_like(theta)
    zeros = torch.zeros_like(theta)

    if axis == "X":
        rows = [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_t, -sin_t], dim=-1),
            torch.stack([zeros, sin_t, cos_t], dim=-1),
        ]
    elif axis == "Y":
        rows = [
            torch.stack([cos_t, zeros, sin_t], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_t, zeros, cos_t], dim=-1),
        ]
    elif axis == "Z":
        rows = [
            torch.stack([cos_t, -sin_t, zeros], dim=-1),
            torch.stack([sin_t, cos_t, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ]
    else:
        raise ValueError("Axis must be X, Y, or Z")

    return torch.stack(rows, dim=-2)  # (..., N, 3, 3)


__all__ = ["xyzeuler_to_hmat"]
