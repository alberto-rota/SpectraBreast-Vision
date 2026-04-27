"""Shared helpers for loading RGB images, GT poses, and GT intrinsics.

These load routines are used by both back-ends and by the top-level
orchestrator. Missing files are handled gracefully so the pipeline can run in
"RGB-only" mode when poses/intrinsics are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose

from .aruco import IMAGE_EXTENSIONS
from .backends import BackendInputs
from .transforms import xyzeuler_to_hmat


def list_rgb_images(rgb_dir: Path) -> List[Path]:
    """Return a sorted list of image paths in a folder."""
    rgb_dir = Path(rgb_dir)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB folder does not exist: {rgb_dir}")
    files = sorted(
        p for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {rgb_dir}")
    return files


def canonicalize_images_with_exif(
    image_paths: List[Path],
    cache_dir: Path,
) -> List[Path]:
    """Bake EXIF orientation into pixel buffers and write to ``cache_dir``.

    iPhones and other cameras often store photos in a fixed sensor orientation
    and use the EXIF ``Orientation`` tag to tell viewers how to rotate/flip
    them for correct display. ``cv2.imread`` / ``torchvision.io.read_image`` /
    ``PIL.Image.open`` all ignore that tag by default, so without this step
    downstream code sees raw buffers that may be upside-down or mirrored.

    Mixed orientations break dense-prediction MASt3R (and similar) models whose
    pose heads assume upright, non-mirrored images. Applying EXIF upfront
    guarantees every consumer (back-end, ArUco detector, Rerun, TUI) works in
    the same image frame.

    The output directory is cleared before writing, and paths are preserved
    in the same sorted order as ``image_paths``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for existing in cache_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    canonical_paths: List[Path] = []
    for src in image_paths:
        # Reserve the sidecar .png extension for everything so we don't have
        # to deal with lossy re-encoding of JPEG->JPEG (which would change
        # pixel values and invalidate ArUco subpixel positions).
        dst = cache_dir / (src.stem + ".png")
        with Image.open(src) as pil_img:
            canonical = exif_transpose(pil_img).convert("RGB")
            canonical.save(dst, format="PNG", compress_level=1)
        canonical_paths.append(dst)
    return canonical_paths


def _fix_3x4_to_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = T
        return out
    raise ValueError(f"Expected (4,4) or (3,4), got {T.shape}")


def load_gt_cameras(
    pose_dir: Path | None,
    camera_params_dir: Path | None,
    num_images: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load GT camera-to-world poses and intrinsics if available.

    Returns ``(T_world_cam, K_orig, cam2ee, distortion)`` where each element
    is ``None`` when the corresponding file is missing.
    """
    T_world_cam: np.ndarray | None = None
    K_orig: np.ndarray | None = None
    cam2ee: np.ndarray | None = None
    distortion: np.ndarray | None = None

    if pose_dir is not None:
        pose_dir = Path(pose_dir)
        if pose_dir.exists():
            pose_files = sorted(pose_dir.glob("pose_*.txt"))
            if len(pose_files) == num_images:
                poses6 = np.stack(
                    [np.asarray(pf.read_text().strip().split(), dtype=np.float32) for pf in pose_files],
                    axis=0,
                )  # [V, 6]
                dev = device if device is not None else torch.device("cpu")
                poses6_t = torch.as_tensor(poses6, dtype=torch.float32, device=dev)
                T_world_ee = xyzeuler_to_hmat(
                    poses6_t,
                    convention="ROLLPITCHYAW",
                    translation_scale=1.0,
                )  # [V, 4, 4]

                if camera_params_dir is not None:
                    cam2ee_path = Path(camera_params_dir) / "camera2ee.npy"
                    if cam2ee_path.exists():
                        cam2ee = _fix_3x4_to_4x4(np.load(cam2ee_path))
                if cam2ee is None:
                    cam2ee = np.eye(4, dtype=np.float32)

                T_ee_cam_t = torch.as_tensor(cam2ee, dtype=torch.float32, device=dev)
                T_ee_cam_batch = T_ee_cam_t.unsqueeze(0).repeat(num_images, 1, 1)
                T_world_cam = torch.matmul(T_world_ee, T_ee_cam_batch).detach().cpu().numpy().astype(np.float32)

    if camera_params_dir is not None:
        cparams = Path(camera_params_dir)
        intrinsics_path = cparams / "intrinsics.npy"
        distortions_path = cparams / "distortions.npy"
        if intrinsics_path.exists():
            K_orig = np.load(intrinsics_path).astype(np.float32)
            if K_orig.shape != (3, 3):
                raise ValueError(f"intrinsics.npy must be [3,3], got {K_orig.shape}")
        if distortions_path.exists():
            distortion = np.load(distortions_path).astype(np.float32).reshape(-1)

    return T_world_cam, K_orig, cam2ee, distortion


def build_backend_inputs(
    rgb_dir: Path,
    pose_dir: Path | None,
    camera_params_dir: Path | None,
    device: torch.device | None = None,
) -> BackendInputs:
    """Gather everything the back-end needs in a single `BackendInputs`."""
    image_paths = list_rgb_images(rgb_dir)
    T_world_cam_gt, K_orig_gt, cam2ee, distortion = load_gt_cameras(
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        num_images=len(image_paths),
        device=device,
    )
    return BackendInputs(
        image_paths=image_paths,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        T_world_cam_gt=T_world_cam_gt,
        K_orig_gt=K_orig_gt,
        cam2ee=cam2ee,
        distortion_coeffs=distortion,
    )


__all__ = [
    "build_backend_inputs",
    "canonicalize_images_with_exif",
    "list_rgb_images",
    "load_gt_cameras",
]
