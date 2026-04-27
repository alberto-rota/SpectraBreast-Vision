"""Camera intrinsic calibration helpers (checkerboard)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _image_paths(image_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_dir.glob(pattern)))
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def calibrate_intrinsics(
    image_dir: Path,
    output_dir: Path = Path("intrinsics"),
    checkerboard_size: tuple[int, int] = (10, 7),
    square_size_m: float = 0.024,
    criteria: tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate intrinsics/distortion from checkerboard images.

    Returns:
        mtx: [3, 3] float64 camera matrix
        dist: [1, K] float64 distortion coefficients (OpenCV model)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    paths = _image_paths(image_dir)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    # object points: [N, 3], N = cols * rows
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints: list[np.ndarray] = []  # each: [N, 3]
    imgpoints: list[np.ndarray] = []  # each: [N, 1, 2]
    image_size: tuple[int, int] | None = None

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (W, H)
        ok, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not ok:
            continue
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)

    if not objpoints or image_size is None:
        raise ValueError(
            "No checkerboard corners found. Check --image-dir, --checkerboard-cols/rows, and board visibility."
        )

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "intrinsics.npy", mtx)
    np.save(output_dir / "distortions.npy", dist)
    return mtx, dist


__all__ = ["calibrate_intrinsics"]
