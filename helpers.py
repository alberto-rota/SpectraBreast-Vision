import torch
import math


def xyzeuler_to_hmat(
    poses: torch.Tensor,
    angles_in_degrees: bool = False,
    convention: str = "XYZ",
    translation_scale: float = 1.0,
) -> torch.Tensor:
    """
    Convert (..., N, 6) XYZ+Euler poses to (..., N, 4, 4) homogeneous transforms.

    Input last dim: [x, y, z, angle1, angle2, angle3]

    Supported conventions:
    - Any 3-character string of 'X', 'Y', 'Z' (e.g., 'XYZ', 'ZYX', 'ZYZ').
      The rotation is applied as R_1(a1) @ R_2(a2) @ R_3(a3).
    - 'RPY' or 'ROLLPITCHYAW': Interprets angles as [Roll, Pitch, Yaw] around X, Y, Z.
      The rotation is applied as R_z(yaw) @ R_y(pitch) @ R_x(roll).

    Args:
        poses: tensor of shape (..., N, 6). Last dim is [x, y, z, a1, a2, a3].
        angles_in_degrees: if True, interpret angles as degrees; else radians.
        convention: String defining the rotation sequence.
        translation_scale: Multiplier for [x, y, z] to convert to desired metric scale.

    Returns:
        Tensor of shape (..., N, 4, 4), same device/dtype as poses.
    """
    assert poses.shape[-1] == 6, "Expected last dim = 6 for [x, y, z, a1, a2, a3]"

    device = poses.device
    dtype = poses.dtype
    shape_prefix = poses.shape[:-1]  # (..., N)

    # Unbind components
    # x, y, z, a1, a2, a3 shapes: (..., N)
    x, y, z, a1, a2, a3 = poses.unbind(dim=-1)

    # Apply translation scale
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
        # a1: Roll (X), a2: Pitch (Y), a3: Yaw (Z)
        # R_axis shapes: (..., N, 3, 3)
        R_x = _axis_angle_to_matrix("X", a1)
        R_y = _axis_angle_to_matrix("Y", a2)
        R_z = _axis_angle_to_matrix("Z", a3)
        # R = R_z @ R_y @ R_x | shape: (..., N, 3, 3)
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    else:
        assert len(conv_upper) == 3 and all(c in "XYZ" for c in conv_upper), (
            f"Invalid convention {convention}"
        )
        R1 = _axis_angle_to_matrix(conv_upper[0], a1)
        R2 = _axis_angle_to_matrix(conv_upper[1], a2)
        R3 = _axis_angle_to_matrix(conv_upper[2], a3)
        # R = R1 @ R2 @ R3 | shape: (..., N, 3, 3)
        R = torch.matmul(R1, torch.matmul(R2, R3))

    # Construct Homogeneous transform (..., N, 4, 4)
    # Initialize as identity
    T = (
        torch.eye(4, dtype=dtype, device=device)
        .view(*[1] * len(shape_prefix), 4, 4)
        .expand(*shape_prefix, 4, 4)
        .clone()
    )  # (..., N, 4, 4)

    # Set rotation block
    T[..., :3, :3] = R

    # Set translation block | shape: (..., N, 3)
    T[..., :3, 3] = torch.stack((x, y, z), dim=-1)

    return T


def _axis_angle_to_matrix(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """
    Generate batch of rotation matrices around a specific axis.

    Args:
        axis: 'X', 'Y', or 'Z'
        theta: Tensor of shape (..., N)

    Returns:
        Tensor of shape (..., N, 3, 3)
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    ones = torch.ones_like(theta)
    zeros = torch.zeros_like(theta)

    if axis == "X":
        # [1, 0, 0], [0, c, -s], [0, s, c]
        rows = [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_t, -sin_t], dim=-1),
            torch.stack([zeros, sin_t, cos_t], dim=-1),
        ]
    elif axis == "Y":
        # [c, 0, s], [0, 1, 0], [-s, 0, c]
        rows = [
            torch.stack([cos_t, zeros, sin_t], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_t, zeros, cos_t], dim=-1),
        ]
    elif axis == "Z":
        # [c, -s, 0], [s, c, 0], [0, 0, 1]
        rows = [
            torch.stack([cos_t, -sin_t, zeros], dim=-1),
            torch.stack([sin_t, cos_t, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ]
    else:
        raise ValueError("Axis must be X, Y, or Z")

    return torch.stack(rows, dim=-2)  # (..., N, 3, 3)