#!/usr/bin/env python3
"""Thin backward-compatible CLI around `spectra.aruco.detect_folder`.

The detection/annotation logic has moved to the importable `spectra.aruco`
module. This script preserves the original command-line interface so
existing workflows keep working.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectra.aruco import ARUCO_DICTIONARIES, detect_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco markers in images from an input folder and write "
            "per-image JSON coordinates plus color-coded annotated output images."
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
        help="Scale factor for marker annotation thickness/text (default: 2.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = detect_folder(
        rgb_dir=args.input_folder,
        out_dir=args.output_folder,
        dictionary=args.aruco_dict,
        draw_scale=max(0.1, args.draw_scale),
    )
    if not results:
        print(f"No images found in: {args.input_folder}")
        return

    for stem, detections in results.items():
        print(f"{stem}: {len(detections)} marker(s)")
    print(f"JSON outputs: {args.output_folder / 'json'}")
    print(f"Annotated images: {args.output_folder / 'annotated'}")


if __name__ == "__main__":
    main()
