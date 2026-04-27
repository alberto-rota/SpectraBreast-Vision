"""Rerun logging helpers shared by the orchestrator.

All the Rerun-specific plumbing lives here so the pipeline logic stays
backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
    HAS_RERUN = True
except Exception:  # pragma: no cover - optional
    rr = None
    rrb = None
    HAS_RERUN = False

from .align import ArucoAlignment, PlaneFrame
from .aruco import MarkerDetection, color_for_id_rgb


def _require_rerun() -> None:
    if not HAS_RERUN:
        raise RuntimeError("rerun is not installed; cannot log visuals.")


def init_rerun(
    app_name: str,
    grpc_port: int,
    *,
    rrd_path: str | Path | None = None,
) -> None:
    """Initialize Rerun; optionally also stream the recording to ``*.rrd`` (multi-sink when supported)."""
    if not HAS_RERUN:
        return
    try:
        rr.disconnect()
    except Exception:
        pass
    rr.init(app_name)
    port = int(grpc_port)
    rrd = Path(rrd_path) if rrd_path is not None else None
    if rrd is not None:
        rrd.parent.mkdir(parents=True, exist_ok=True)
        p = str(rrd)
        gurl = f"rerun+http://127.0.0.1:{port}/proxy"
        set_sinks = getattr(rr, "set_sinks", None)
        if callable(set_sinks) and hasattr(rr, "FileSink") and hasattr(rr, "GrpcSink"):
            last_exc: Exception | None = None
            for factory in (
                (lambda: (rr.GrpcSink(url=gurl), rr.FileSink(p))),
                (lambda: (rr.GrpcSink(), rr.FileSink(p))),
            ):
                try:
                    g, f = factory()
                    set_sinks(g, f)
                    return
                except Exception as exc:  # pragma: no cover - version-matrix
                    last_exc = exc
            from rich import print as rprint  # type: ignore[import-not-found]

            rprint(
                f"[yellow]Rerun: could not open .rrd FileSink ({last_exc!r}); gRPC only (no {p}).[/yellow]"
            )
        else:
            from rich import print as rprint  # type: ignore[import-not-found]

            rprint(
                f"[yellow]Rerun: set_sinks / FileSink / GrpcSink missing; gRPC only (no {p}).[/yellow]"
            )
    rr.serve_grpc(grpc_port=port)


def _save_blueprint_file(
    blueprint: "rrb.Blueprint", application_id: str, rbl_path: str | Path
) -> None:
    """Write layout to ``*.rbl``; ``application_id`` must match :func:`init_rerun` app name."""
    if not HAS_RERUN:
        return
    p = Path(rbl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save = getattr(blueprint, "save", None)
    if not callable(save):
        return
    try:
        save(application_id, str(p))  # type: ignore[call-arg]
    except TypeError:  # pragma: no cover - API drift
        try:
            save(str(p), application_id=application_id)  # type: ignore[call-arg]
        except (TypeError, OSError) as e:
            from rich import print as rprint  # type: ignore[import-not-found]

            rprint(f"[yellow]Rerun: could not save .rbl ({e}).[/yellow]")


def _spatial3d_view_content_query(path: str) -> str:
    """Build a Rerun 0.2+ :class:`Spatial3DView` *contents* query (see ViewContents).

    Plain path strings like ``"/aruco"`` match at most that *single* entity. ArUco
    data is logged on **children** (e.g. ``/aruco/0/face``, ``/aruco/plane``), so
    the Scene view would show no markers. Hierarchical roots need a subtree
    include such as ``"+ /aruco/**"`` (same for ``/cameras`` and ``/surface``).
    """
    p = path if path.startswith("/") else f"/{path}"
    if p in ("/aruco", "/cameras", "/surface"):
        return f"+ {p}/**"
    return f"+ {p}"


def send_blueprint(
    *,
    include_scene_view: bool = True,
    include_confidence_view: bool = True,
    include_camera_images_view: bool = True,
    include_cameras: bool = True,
    include_cloud: bool = True,
    include_cloud_confidence: bool = True,
    include_aruco: bool = True,
    include_surface: bool = True,
    application_id: str | None = None,
    rbl_path: str | Path | None = None,
) -> None:
    if not HAS_RERUN:
        return
    top_row_views = []
    if include_scene_view:
        scene_contents = []
        if include_cloud:
            scene_contents.append(_spatial3d_view_content_query("/points"))
        if include_cameras:
            scene_contents.append(_spatial3d_view_content_query("/cameras"))
        if include_aruco:
            scene_contents.append(_spatial3d_view_content_query("/aruco"))
        if include_surface:
            scene_contents.append(_spatial3d_view_content_query("/surface"))
        if scene_contents:
            top_row_views.append(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/",
                    contents=scene_contents,
                )
            )

    if include_confidence_view and include_cloud_confidence:
        top_row_views.append(
            rrb.Spatial3DView(
                name="Confidence",
                origin="/",
                contents=[_spatial3d_view_content_query("/points_confidence")],
            )
        )

    layout_children = []
    if top_row_views:
        if len(top_row_views) == 1:
            layout_children.append(top_row_views[0])
        else:
            layout_children.append(rrb.Horizontal(*top_row_views))

    if include_camera_images_view and include_cameras:
        layout_children.append(rrb.Spatial2DView(name="Camera images", origin="/cameras"))

    if not layout_children:
        return
    if len(layout_children) == 1:
        blueprint = rrb.Blueprint(layout_children[0])
    else:
        blueprint = rrb.Blueprint(rrb.Vertical(*layout_children))
    rr.send_blueprint(blueprint, make_active=True)
    if rbl_path is not None and application_id is not None:
        _save_blueprint_file(blueprint, application_id, rbl_path)


def rerun_flush() -> None:
    if not HAS_RERUN:
        return
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
    except Exception:
        pass


def log_camera(
    index: int,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    resolution_wh: Sequence[int],
    image_rgb: np.ndarray | None = None,
    image_aruco_rgb: np.ndarray | None = None,
    confidence_rgb: np.ndarray | None = None,
    image_plane_distance: float = 0.01,
) -> None:
    if not HAS_RERUN:
        return
    rr.log(
        f"/cameras/{index}",
        rr.Transform3D(translation=T_world_cam[:3, 3], mat3x3=T_world_cam[:3, :3]),
    )
    pinhole_kwargs: dict = {
        "image_from_camera": np.asarray(K, dtype=np.float32),
        "resolution": list(resolution_wh),
        "image_plane_distance": image_plane_distance,
    }
    # Match OpenCV / VGGT (X right, Y down, Z forward) so frustums line up with ``/points`` + ``/aruco``.
    vc = getattr(rr, "ViewCoordinates", None)
    if vc is not None and hasattr(vc, "RDF"):
        pinhole_kwargs["camera_xyz"] = vc.RDF
    rr.log(f"/cameras/{index}/image", rr.Pinhole(**pinhole_kwargs))
    if image_rgb is not None:
        rr.log(f"/cameras/{index}/image/rgb", rr.Image(image_rgb))
    if image_aruco_rgb is not None:
        rr.log(f"/cameras/{index}/image/aruco", rr.Image(image_aruco_rgb))
    if confidence_rgb is not None:
        rr.log(f"/cameras/{index}/image/confidence", rr.Image(confidence_rgb))


def log_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    conf_colors: np.ndarray | None = None,
    *,
    log_rgb: bool = True,
    log_confidence: bool = True,
) -> None:
    if not HAS_RERUN:
        return
    if log_rgb:
        rr.log("/points", rr.Points3D(points, colors=colors), static=True)
    if log_confidence and conf_colors is not None:
        rr.log("/points_confidence", rr.Points3D(points, colors=conf_colors), static=True)


def resample_cloud_for_logging(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample cloud consistently across points/colors/confidence."""
    factor = float(factor)
    if factor >= 1.0 or points.shape[0] <= 1:
        return points, colors, confidence
    keep_count = max(1, int(round(points.shape[0] * factor)))
    keep_idx = np.linspace(0, points.shape[0] - 1, num=keep_count, dtype=np.int64)
    return points[keep_idx], colors[keep_idx], confidence[keep_idx]


def _log_marker_highlight_3d(
    marker_id: int,
    corners_3d: np.ndarray,  # shape: (4, 3), float
    path_prefix: str = "/aruco",
    face_alpha: int = 180,
) -> None:
    """Log one ArUco marker as a filled colored quad + outline + corners + ID label + normal arrow.

    Sizes (corner radii, line thickness, normal length) are scaled to the
    marker's own edge length so the visualization looks correct whether we're
    in meters, millimeters, or arbitrary back-end units.
    """
    if not HAS_RERUN:
        return

    corners_3d = np.asarray(corners_3d, dtype=np.float32).reshape(4, 3)  # (4, 3)
    color_rgb = np.asarray(color_for_id_rgb(marker_id), dtype=np.uint8)   # (3,)
    color_rgba = np.concatenate([color_rgb, np.uint8([face_alpha])])      # (4,)

    # Edge length = mean of 4 side lengths (vectorized, no explicit loop).
    edges_vec = np.roll(corners_3d, -1, axis=0) - corners_3d              # (4, 3)
    edge_lengths = np.linalg.norm(edges_vec, axis=1)                      # (4,)
    edge_len = float(edge_lengths.mean())
    if not np.isfinite(edge_len) or edge_len <= 0.0:
        edge_len = 0.01

    center = corners_3d.mean(axis=0)                                      # (3,)
    corner_radius = max(0.06 * edge_len, 1e-4)
    line_radius = max(0.04 * edge_len, 1e-4)
    label_radius = max(0.10 * edge_len, 1e-4)

    # Filled marker face (two triangles forming the quad).
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)       # (2, 3)
    face_colors = np.tile(color_rgba, (4, 1))                             # (4, 4)
    rr.log(
        f"{path_prefix}/{int(marker_id)}/face",
        rr.Mesh3D(
            vertex_positions=corners_3d,
            triangle_indices=triangles,
            vertex_colors=face_colors,
        ),
        static=True,
    )

    # Thick colored outline.
    closed = np.concatenate([corners_3d, corners_3d[:1]], axis=0)         # (5, 3)
    rr.log(
        f"{path_prefix}/{int(marker_id)}/edges",
        rr.LineStrips3D([closed], colors=[color_rgb], radii=[line_radius]),
        static=True,
    )

    # Larger corner spheres.
    corner_colors = np.tile(color_rgb, (4, 1))                            # (4, 3)
    rr.log(
        f"{path_prefix}/{int(marker_id)}/corners",
        rr.Points3D(corners_3d, colors=corner_colors, radii=corner_radius),
        static=True,
    )

    # ID label at the marker center.
    rr.log(
        f"{path_prefix}/{int(marker_id)}/label",
        rr.Points3D(
            center[None, :],
            colors=[color_rgb],
            radii=[label_radius],
            labels=[f"id:{int(marker_id)}"],
        ),
        static=True,
    )

    # Outward normal arrow: n = (c1 - c0) x (c3 - c0), length = edge.
    e_u = corners_3d[1] - corners_3d[0]
    e_v = corners_3d[3] - corners_3d[0]
    normal = np.cross(e_u, e_v)
    nlen = float(np.linalg.norm(normal))
    if nlen > 1e-9:
        normal = (normal / nlen).astype(np.float32)
        rr.log(
            f"{path_prefix}/{int(marker_id)}/normal",
            rr.Arrows3D(
                origins=center[None, :],
                vectors=(normal * edge_len)[None, :],
                colors=[color_rgb],
                radii=[line_radius * 0.8],
            ),
            static=True,
        )


def log_aruco_triangulations(alignment: ArucoAlignment) -> None:
    """Highlight triangulated ArUco markers in Rerun (per-ID color, face + edges + label + normal)."""
    if not HAS_RERUN:
        return
    for marker_id, marker in alignment.markers.items():
        _log_marker_highlight_3d(marker_id, marker.corners_3d)


def log_xy_plane(plane_frame: PlaneFrame, size_m: float = 0.1) -> None:
    if not HAS_RERUN or plane_frame is None:
        return
    rr.log(
        "/aruco/plane",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[[float(size_m), float(size_m), 1e-4]],
            colors=[[180, 180, 255]],
        ),
        static=True,
    )


def log_aruco_alignment_in_output_frame(
    aligned_corners_3d: Mapping[int, np.ndarray],
    size_m: float = 0.1,
) -> None:
    """Highlight aligned ArUcos in the output frame + the Z=0 plane rectangle."""
    if not HAS_RERUN:
        return
    for marker_id, corners_3d in aligned_corners_3d.items():
        _log_marker_highlight_3d(marker_id, corners_3d)

    if aligned_corners_3d:
        all_xy = np.concatenate([c[:, :2] for c in aligned_corners_3d.values()], axis=0)
        span = float(max(np.ptp(all_xy, axis=0).max(), size_m))
    else:
        span = size_m
    rr.log(
        "/aruco/plane",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[[span, span, 1e-4]],
            colors=[[180, 180, 255, 90]],
        ),
        static=True,
    )


def log_surface_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> None:
    if not HAS_RERUN or vertices.shape[0] == 0 or triangles.shape[0] == 0:
        return
    kwargs = {
        "vertex_positions": np.asarray(vertices, dtype=np.float32),
        "triangle_indices": np.asarray(triangles, dtype=np.uint32),
    }
    if vertex_colors is not None:
        kwargs["vertex_colors"] = np.asarray(vertex_colors, dtype=np.uint8)
    if vertex_normals is not None:
        kwargs["vertex_normals"] = np.asarray(vertex_normals, dtype=np.float32)
    rr.log("/surface/mesh", rr.Mesh3D(**kwargs), static=True)


__all__ = [
    "HAS_RERUN",
    "init_rerun",
    "log_aruco_alignment_in_output_frame",
    "log_aruco_triangulations",
    "log_camera",
    "log_cloud",
    "log_surface_mesh",
    "log_xy_plane",
    "resample_cloud_for_logging",
    "rerun_flush",
    "send_blueprint",
]
