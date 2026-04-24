"""ArUco marker detection with per-ID deterministic colors.

This module exposes a small, import-friendly API for detecting ArUco markers
in images:

- `ArucoDetector` — stateful detector wrapping `cv2.aruco.ArucoDetector`
- `detect_image(image_bgr, detector)` — detect markers in a single BGR image
- `detect_folder(rgb_dir, out_dir, ...)` — batch detection + optional disk
  outputs (JSON + color-coded annotated images)

It also provides `color_for_id()` which yields a stable distinct BGR color
per marker ID, used consistently by annotations and 3D visualizations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


ARUCO_DICTIONARIES: Dict[str, int] = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class MarkerDetection:
    """A single ArUco marker detected in a 2D image.

    `corners_xy` follows OpenCV's convention (top-left, top-right,
    bottom-right, bottom-left) in image pixel coordinates.
    """

    id: int
    corners_xy: np.ndarray  # shape: (4, 2), float32
    center_xy: np.ndarray   # shape: (2,), float32

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "corners_xy": np.round(self.corners_xy, 3).tolist(),
            "center_xy": np.round(self.center_xy, 3).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MarkerDetection":
        corners_xy = np.asarray(payload["corners_xy"], dtype=np.float32).reshape(4, 2)
        center_xy = np.asarray(payload["center_xy"], dtype=np.float32).reshape(2)
        return cls(id=int(payload["id"]), corners_xy=corners_xy, center_xy=center_xy)


def color_for_id(marker_id: int) -> tuple[int, int, int]:
    """Return a stable, vivid BGR color for a given marker ID.

    Uses golden-ratio hue sampling so consecutive IDs land far apart in HSV
    space and every ID gets a visually distinct color.
    """
    golden_ratio_conjugate = 0.6180339887498949
    hue = (int(marker_id) * golden_ratio_conjugate) % 1.0
    # Full saturation and high value for vivid, easy-to-read colors.
    hsv = np.array([[[hue * 179.0, 255.0, 230.0]]], dtype=np.float32)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def color_for_id_rgb(marker_id: int) -> tuple[int, int, int]:
    """RGB variant of `color_for_id` (convenient for Rerun / matplotlib)."""
    b, g, r = color_for_id(marker_id)
    return (r, g, b)


class ArucoDetector:
    """Stateful OpenCV ArUco detector with a clean Python API."""

    def __init__(
        self,
        dictionary: str = "4x4_50",
        parameters: cv2.aruco.DetectorParameters | None = None,
    ) -> None:
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(
                f"Unknown ArUco dictionary {dictionary!r}. "
                f"Valid options: {sorted(ARUCO_DICTIONARIES.keys())}"
            )
        self.dictionary_name = dictionary
        self._dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
        self._parameters = parameters if parameters is not None else cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._parameters)

    def detect(self, image_bgr: np.ndarray) -> List[MarkerDetection]:
        """Detect ArUco markers in a single BGR image."""
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        # corners is a tuple of [1, 4, 2] float32 arrays; stack to [N, 4, 2].
        marker_corners = np.squeeze(np.asarray(corners, dtype=np.float32), axis=1)  # [N, 4, 2]
        marker_ids = ids.reshape(-1).astype(int)  # [N]
        centers = marker_corners.mean(axis=1)  # [N, 2]

        detections: List[MarkerDetection] = []
        for idx in range(len(marker_ids)):
            detections.append(
                MarkerDetection(
                    id=int(marker_ids[idx]),
                    corners_xy=marker_corners[idx].astype(np.float32),
                    center_xy=centers[idx].astype(np.float32),
                )
            )
        return detections


def detect_image(
    image_bgr: np.ndarray,
    detector: ArucoDetector | None = None,
    dictionary: str = "4x4_50",
) -> List[MarkerDetection]:
    """Convenience wrapper: detect markers in a single image."""
    if detector is None:
        detector = ArucoDetector(dictionary=dictionary)
    return detector.detect(image_bgr)


def _draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[MarkerDetection],
    draw_scale: float = 2.0,
) -> np.ndarray:
    """Draw detections with per-ID stable colors on a copy of the image."""
    annotated = image_bgr.copy()
    image_h, image_w = annotated.shape[:2]
    resolution_scale = max(image_h, image_w) / 1280.0
    visual_scale = max(0.5, draw_scale * resolution_scale)

    line_thickness = max(2, int(round(3.0 * visual_scale)))
    circle_radius = max(3, int(round(6.0 * visual_scale)))
    text_scale = max(0.7, 0.9 * visual_scale)
    text_thickness = max(2, int(round(2.0 * visual_scale)))

    for detection in detections:
        color = color_for_id(detection.id)
        pts = detection.corners_xy.astype(np.int32)

        cv2.polylines(
            annotated,
            [pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        center_xy = detection.center_xy.astype(np.int32)
        cv2.circle(
            annotated,
            tuple(center_xy.tolist()),
            circle_radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        for corner_xy in pts:
            cv2.circle(
                annotated,
                tuple(corner_xy.tolist()),
                max(2, circle_radius - 1),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        label = f"id:{detection.id}"
        text_anchor = (int(pts[0, 0]), max(20, int(pts[0, 1]) - 12))
        cv2.putText(
            annotated,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            thickness=max(3, text_thickness + 2),
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def annotate_image(
    image_bgr: np.ndarray,
    detections: Sequence[MarkerDetection],
    draw_scale: float = 2.0,
) -> np.ndarray:
    """Public alias for the annotation drawer."""
    return _draw_detections(image_bgr, detections, draw_scale=draw_scale)


def list_images(input_folder: Path) -> List[Path]:
    return sorted(
        p
        for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_detections_json(path: Path, image_name: str, detections: Sequence[MarkerDetection]) -> None:
    payload = {
        "image_name": image_name,
        "num_detections": len(detections),
        "detections": [d.to_dict() for d in detections],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_detections_json(path: Path) -> List[MarkerDetection]:
    """Load a per-image detections JSON produced by `detect_folder`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MarkerDetection.from_dict(d) for d in payload.get("detections", [])]


def detect_folder(
    rgb_dir: Path,
    out_dir: Path | None = None,
    dictionary: str = "4x4_50",
    draw_scale: float = 2.0,
    image_paths: Iterable[Path] | None = None,
) -> Dict[str, List[MarkerDetection]]:
    """Run ArUco detection on every image in a folder.

    If `out_dir` is provided, per-image JSON files go into
    `out_dir/json/` and color-coded annotated PNGs into
    `out_dir/annotated/`.

    Returns a dict mapping image stem -> list of detections, with the
    image iteration order preserved (sorted by filename).
    """
    rgb_dir = Path(rgb_dir)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {rgb_dir}")

    detector = ArucoDetector(dictionary=dictionary)

    if image_paths is None:
        images = list_images(rgb_dir)
    else:
        images = [Path(p) for p in image_paths]
    if not images:
        return {}

    json_dir: Path | None = None
    annotated_dir: Path | None = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        json_dir = out_dir / "json"
        annotated_dir = out_dir / "annotated"
        json_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[MarkerDetection]] = {}
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            results[image_path.stem] = []
            continue

        detections = detector.detect(image)
        results[image_path.stem] = detections

        if json_dir is not None and annotated_dir is not None:
            _write_detections_json(json_dir / f"{image_path.stem}.json", image_path.name, detections)
            annotated = _draw_detections(image, detections, draw_scale=draw_scale)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    return results


__all__ = [
    "ARUCO_DICTIONARIES",
    "ArucoDetector",
    "IMAGE_EXTENSIONS",
    "MarkerDetection",
    "annotate_image",
    "color_for_id",
    "color_for_id_rgb",
    "detect_folder",
    "detect_image",
    "list_images",
    "read_detections_json",
]
