#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco markers in images from an input folder and write "
            "per-image JSON coordinates plus annotated output images."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to folder containing input images.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help=(
            "Path to output folder. Script creates 'json/' and 'annotated/' "
            "subfolders inside it."
        ),
    )
    parser.add_argument(
        "--dict",
        dest="aruco_dict",
        type=str,
        default="4x4_50",
        choices=sorted(ARUCO_DICTIONARIES.keys()),
        help="ArUco dictionary name (default: 4x4_50).",
    )
    parser.add_argument(
        "--draw-scale",
        type=float,
        default=2.0,
        help="Scale factor for marker annotations thickness/text (default: 2.0).",
    )
    return parser.parse_args()


def list_images(input_folder: Path) -> List[Path]:
    return sorted(
        [
            image_path
            for image_path in input_folder.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def marker_record(marker_id: int, corners_xy: np.ndarray) -> Dict:
    # corners_xy shape: (4, 2), with each row [x, y] in image pixel space
    center_xy = corners_xy.mean(axis=0)  # shape: (2,)
    return {
        "id": marker_id,
        "corners_xy": corners_xy.round(3).tolist(),
        "center_xy": center_xy.round(3).tolist(),
    }


def draw_marker_annotations(
    image_bgr: np.ndarray,
    marker_ids: np.ndarray,
    marker_corners: np.ndarray,
    draw_scale: float,
) -> np.ndarray:
    image_h, image_w = image_bgr.shape[:2]
    resolution_scale = max(image_h, image_w) / 1280.0
    visual_scale = max(0.5, draw_scale * resolution_scale)

    line_thickness = max(2, int(round(3.0 * visual_scale)))
    circle_radius = max(3, int(round(6.0 * visual_scale)))
    text_scale = max(0.7, 0.9 * visual_scale)
    text_thickness = max(2, int(round(2.0 * visual_scale)))

    for marker_id, corners_xy in zip(marker_ids.tolist(), marker_corners):
        pts = corners_xy.astype(np.int32)
        cv2.polylines(
            image_bgr,
            [pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        center_xy = corners_xy.mean(axis=0).astype(np.int32)
        cv2.circle(
            image_bgr,
            tuple(center_xy.tolist()),
            circle_radius,
            (0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        for corner_xy in pts:
            cv2.circle(
                image_bgr,
                tuple(corner_xy.tolist()),
                max(2, circle_radius - 1),
                (255, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        label = f"id:{marker_id}"
        text_anchor = (int(pts[0, 0]), max(20, int(pts[0, 1]) - 12))
        cv2.putText(
            image_bgr,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            thickness=max(3, text_thickness + 2),
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return image_bgr


def detect_markers(
    image_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    draw_scale: float,
) -> Tuple[List[Dict], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    annotated = image_bgr.copy()
    detections: List[Dict] = []

    if ids is None or len(ids) == 0:
        return detections, annotated

    marker_ids = ids.reshape(-1).astype(int)
    marker_corners = np.squeeze(np.asarray(corners, dtype=np.float32), axis=1)
    annotated = draw_marker_annotations(annotated, marker_ids, marker_corners, draw_scale)
    for marker_id, corners_xy in zip(marker_ids.tolist(), marker_corners):
        detections.append(marker_record(marker_id, corners_xy))

    return detections, annotated


def write_json(path: Path, image_name: str, detections: List[Dict]) -> None:
    payload = {
        "image_name": image_name,
        "num_detections": len(detections),
        "detections": detections,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_folder}")

    json_folder = output_folder / "json"
    annotated_folder = output_folder / "annotated"
    json_folder.mkdir(parents=True, exist_ok=True)
    annotated_folder.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[args.aruco_dict])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    image_paths = list_images(input_folder)
    if not image_paths:
        print(f"No images found in: {input_folder}")
        return

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        detections, annotated = detect_markers(
            image,
            detector,
            draw_scale=max(0.1, args.draw_scale),
        )

        json_path = json_folder / f"{image_path.stem}.json"
        write_json(json_path, image_path.name, detections)

        annotated_path = annotated_folder / image_path.name
        cv2.imwrite(str(annotated_path), annotated)

        print(f"{image_path.name}: {len(detections)} marker(s)")

    print(f"JSON outputs: {json_folder}")
    print(f"Annotated images: {annotated_folder}")


if __name__ == "__main__":
    main()
