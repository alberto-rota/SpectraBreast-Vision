"""Shared I/O helpers for point clouds and meshes (ASCII PLY format)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def save_pointcloud_as_ply(
    path: Path,
    points: np.ndarray,           # [N, 3]
    colors: np.ndarray,           # [N, 3] uint8
    confidence: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    support: np.ndarray | None = None,
) -> Path:
    """Write a colored point cloud to an ASCII PLY file.

    Optionally stores per-vertex confidence, normals, and support counts as
    extra vertex properties.
    """
    path = Path(path)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points ({points.shape}) and colors ({colors.shape}) must have the same N"
        )
    if confidence is not None and confidence.shape[0] != points.shape[0]:
        raise ValueError(f"confidence must have shape [{points.shape[0]}], got {confidence.shape}")
    if normals is not None and normals.shape != points.shape:
        raise ValueError(f"normals must match points shape {points.shape}, got {normals.shape}")
    if support is not None and support.shape[0] != points.shape[0]:
        raise ValueError(f"support must have shape [{points.shape[0]}], got {support.shape}")

    num_points = int(points.shape[0])
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
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    header.extend([
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ])
    if confidence is not None:
        header.append("property float confidence")
    if support is not None:
        header.append("property float support")
    header.append("end_header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if num_points == 0:
            return path

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
    return path


def save_mesh_as_ply(
    path: Path,
    vertices: np.ndarray,                # [Nv, 3]
    triangles: np.ndarray,               # [Nt, 3] int32
    vertex_colors: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> Path:
    """Write a triangle mesh (vertices + faces) to an ASCII PLY file."""
    path = Path(path)
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape [N,3], got {verts.shape}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"Expected triangles with shape [M,3], got {tris.shape}")
    if vertex_colors is not None and vertex_colors.shape != verts.shape:
        raise ValueError(f"Expected vertex_colors shape {verts.shape}, got {vertex_colors.shape}")
    if vertex_normals is not None and vertex_normals.shape != verts.shape:
        raise ValueError(f"Expected vertex_normals shape {verts.shape}, got {vertex_normals.shape}")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {verts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if vertex_normals is not None:
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    if vertex_colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header.extend([
        f"element face {tris.shape[0]}",
        "property list uchar int vertex_indices",
        "end_header",
    ])

    vertex_parts = [verts]
    vertex_fmt = ["%.6f", "%.6f", "%.6f"]
    if vertex_normals is not None:
        vertex_parts.append(vertex_normals.astype(np.float32))
        vertex_fmt.extend(["%.6f", "%.6f", "%.6f"])
    if vertex_colors is not None:
        vertex_parts.append(vertex_colors.astype(np.float32))
        vertex_fmt.extend(["%d", "%d", "%d"])
    vertex_data = np.concatenate(vertex_parts, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
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
    return path


def prepare_run_dir(root: Path, run_name: str | None = None) -> tuple[Path, str]:
    """Create a timestamped run folder and return (path, run_name)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_dir.name


def update_most_recent_symlink(root: Path, run_dir: Path) -> Path:
    """Point `<root>/most-recent` at `run_dir` (falls back to a copy on failure)."""
    root = Path(root)
    target = root / "most-recent"
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    try:
        target.symlink_to(run_dir.relative_to(root), target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, target)
    return target


__all__ = [
    "prepare_run_dir",
    "save_mesh_as_ply",
    "save_pointcloud_as_ply",
    "update_most_recent_symlink",
]
