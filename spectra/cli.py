"""Typer CLI: MASt3R-SfM reconstruction + ArUco 2D/3D, optional GT poses/intrinsics.

After ``pip install -e .`` from the repo root, invoke as ``spectra``.
Equivalent: ``python -m spectra``.

Usage::

    # Config-driven (see configs/default.yaml)
    spectra run --config configs/default.yaml

    # Minimal: images only, outputs under RESULTS/
    spectra run --rgb-dir path/to/images

    # With optional calibration (pose_*.txt per view, intrinsics.npy, …)
    spectra run -c configs/default.yaml --pose-dir DATA/poses --camera-params-dir DATA/cam

    # 2D ArUco detection only (no SfM)
    spectra detect path/to/rgb /tmp/out

    # Local 3D viewer (optional gradio)
    spectra viewer -r RESULTS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from .aruco import ARUCO_DICTIONARIES, detect_folder
from .calibration import calibrate_intrinsics
from .config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    ReconstructionConfig,
    RerunConfig,
    SurfaceConfig,
    load_config,
)


def _parse_override(value: str) -> tuple[str, object]:
    """Parse ``key.path=value``; value parsed with ``json.loads`` when possible."""
    if "=" not in value:
        raise typer.BadParameter(f"Override {value!r} must be in the form 'a.b=VALUE'")
    dotted_key, raw_value = value.split("=", 1)
    dotted_key = dotted_key.strip()
    raw_value = raw_value.strip()
    try:
        parsed: object = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value
    return dotted_key, parsed


def _build_config(
    config: Optional[Path],
    rgb_dir: Optional[Path],
    pose_dir: Optional[Path],
    camera_params_dir: Optional[Path],
    out_dir: Optional[Path],
    run_name: Optional[str],
    marker_edge_length_m: Optional[float],
    aruco_dictionary: Optional[str],
    align_to_aruco: Optional[bool],
    origin_marker_id: Optional[int],
    grpc_port: Optional[int],
    no_wait: bool,
    overrides: List[str],
) -> ReconstructionConfig:
    if config is not None:
        cfg = load_config(config)
    else:
        if rgb_dir is None:
            raise typer.BadParameter("Pass --config or --rgb-dir with a folder of images.")
        cfg = ReconstructionConfig(
            input=InputConfig(rgb_dir=rgb_dir, pose_dir=pose_dir, camera_params_dir=camera_params_dir),
            output=OutputConfig(),
            aruco=ArucoConfig(),
            surface=SurfaceConfig(),
            mast3r=Mast3rConfig(),
            rerun=RerunConfig(),
        )

    direct: dict[str, object] = {}
    if rgb_dir is not None:
        direct["input.rgb_dir"] = str(rgb_dir)
    if pose_dir is not None:
        direct["input.pose_dir"] = str(pose_dir)
    if camera_params_dir is not None:
        direct["input.camera_params_dir"] = str(camera_params_dir)
    if out_dir is not None:
        direct["output.root"] = str(out_dir)
    if run_name is not None:
        direct["output.run_name"] = run_name
    if marker_edge_length_m is not None:
        direct["aruco.marker_edge_length_m"] = float(marker_edge_length_m)
    if aruco_dictionary is not None:
        direct["aruco.dictionary"] = aruco_dictionary
    if align_to_aruco is not None:
        direct["aruco.align_to_aruco"] = bool(align_to_aruco)
    if origin_marker_id is not None:
        direct["aruco.origin_marker_id"] = int(origin_marker_id)
    if grpc_port is not None:
        direct["rerun.grpc_port"] = int(grpc_port)
    if no_wait:
        direct["rerun.no_wait"] = True

    if direct:
        cfg = cfg.with_overrides(direct)

    if overrides:
        merged: dict[str, object] = {}
        for ov in overrides:
            key, value = _parse_override(ov)
            merged[key] = value
        cfg = cfg.with_overrides(merged)
    return cfg


def _execute_run(
    config: Optional[Path],
    rgb_dir: Optional[Path],
    pose_dir: Optional[Path],
    camera_params_dir: Optional[Path],
    out_dir: Optional[Path],
    run_name: Optional[str],
    marker_edge_length_m: Optional[float],
    aruco_dictionary: Optional[str],
    align_to_aruco: Optional[bool],
    origin_marker_id: Optional[int],
    grpc_port: Optional[int],
    no_wait: bool,
    overrides: List[str],
) -> None:
    cfg = _build_config(
        config=config,
        rgb_dir=rgb_dir,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        out_dir=out_dir,
        run_name=run_name,
        marker_edge_length_m=marker_edge_length_m,
        aruco_dictionary=aruco_dictionary,
        align_to_aruco=align_to_aruco,
        origin_marker_id=origin_marker_id,
        grpc_port=grpc_port,
        no_wait=no_wait,
        overrides=overrides,
    )
    from .pipeline import run_reconstruction

    result = run_reconstruction(cfg)
    print(f"[green]Run directory:[/green] {result.run_dir}")


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="MASt3R-SfM + ArUco: images → point cloud, markers in 2D/3D.",
)


@app.command("recon")
def recon_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="YAML config (recommended: configs/default.yaml).",
    ),
    rgb_dir: Optional[Path] = typer.Option(None, "--rgb-dir", help="Input images folder."),
    pose_dir: Optional[Path] = typer.Option(
        None, "--pose-dir", help="Optional GT poses (pose_*.txt per view)."
    ),
    camera_params_dir: Optional[Path] = typer.Option(
        None, "--camera-params-dir", help="Optional intrinsics.npy / camera2ee.npy directory."
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output root (default from config or RESULTS)."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Subfolder name under output root."),
    marker_edge_length_m: Optional[float] = typer.Option(
        None, "--marker-m", help="ArUco physical edge length (meters), overrides config."
    ),
    aruco_dictionary: Optional[str] = typer.Option(None, "--dict", help="OpenCV ArUco dictionary, e.g. 4x4_50."),
    align_to_aruco: Optional[bool] = typer.Option(
        None, "--align/--no-align", help="Sim3 align dense cloud to ArUco (default: config).",
    ),
    origin_marker_id: Optional[int] = typer.Option(
        None, "--origin-id", help="If set, this marker defines the XY origin."
    ),
    grpc_port: Optional[int] = typer.Option(None, "--grpc-port", help="Rerun gRPC port."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Do not block after Rerun logging."),
    set_override: List[str] = typer.Option(
        [],
        "--set",
        "-s",
        help="YAML override: --set aruco.marker_edge_length_m=0.09",
    ),
) -> None:
    """Run MASt3R-SfM and export fused cloud + ArUco 2D/3D."""
    _execute_run(
        config=config,
        rgb_dir=rgb_dir,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        out_dir=out_dir,
        run_name=run_name,
        marker_edge_length_m=marker_edge_length_m,
        aruco_dictionary=aruco_dictionary,
        align_to_aruco=align_to_aruco,
        origin_marker_id=origin_marker_id,
        grpc_port=grpc_port,
        no_wait=no_wait,
        overrides=set_override,
    )


@app.command("reconstruct", hidden=True)
def reconstruct_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        readable=True,
    ),
    rgb_dir: Optional[Path] = typer.Option(None, "--rgb-dir"),
    pose_dir: Optional[Path] = typer.Option(None, "--pose-dir"),
    camera_params_dir: Optional[Path] = typer.Option(None, "--camera-params-dir"),
    out_dir: Optional[Path] = typer.Option(
        None, "--out", "--out-dir", "-o", help="Output root; --out-dir is a legacy alias.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    marker_edge_length_m: Optional[float] = typer.Option(None, "--marker-edge-length-m"),
    aruco_dictionary: Optional[str] = typer.Option(None, "--aruco-dictionary"),
    align_to_aruco: Optional[bool] = typer.Option(None, "--align-to-aruco/--no-align-to-aruco"),
    origin_marker_id: Optional[int] = typer.Option(None, "--origin-marker-id"),
    grpc_port: Optional[int] = typer.Option(None, "--grpc-port"),
    no_wait: bool = False,
    set_override: List[str] = typer.Option([], "--set", "-s"),
) -> None:
    """Alias for ``spectra run`` (deprecated name)."""
    _execute_run(
        config=config,
        rgb_dir=rgb_dir,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        out_dir=out_dir,
        run_name=run_name,
        marker_edge_length_m=marker_edge_length_m,
        aruco_dictionary=aruco_dictionary,
        align_to_aruco=align_to_aruco,
        origin_marker_id=origin_marker_id,
        grpc_port=grpc_port,
        no_wait=no_wait,
        overrides=set_override,
    )


@app.command("detect")
def detect_cmd(
    input_folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of RGB images."),
    output_folder: Path = typer.Argument(
        ..., help="Receives json/ and annotated/ (parent is created as needed).",
    ),
    dictionary: str = typer.Option(
        "4x4_50",
        "--dict",
        help=f"ArUco dictionary. One of: {sorted(ARUCO_DICTIONARIES.keys())}",
    ),
    draw_scale: float = typer.Option(
        2.0, "--draw-scale", help="Annotation line thickness / text size multiplier."
    ),
) -> None:
    """Detect ArUco markers in images (2D only; no 3D / no SfM)."""
    results = detect_folder(
        rgb_dir=input_folder,
        out_dir=output_folder,
        dictionary=dictionary,
        draw_scale=max(0.1, draw_scale),
    )
    if not results:
        print(f"[yellow]No images found in:[/yellow] {input_folder}")
        return

    for stem, detections in results.items():
        print(f"{stem}: [green]{len(detections)}[/green] marker(s)")
    print(f"JSON: [green]{output_folder / 'json'}[/green]")
    print(f"Annotated: [green]{output_folder / 'annotated'}[/green]")


@app.command("tui", hidden=True)
def tui_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
    ),
    data_root: Path = typer.Option(Path("DATA"), "--data-root"),
) -> None:
    try:
        from .tui import run_tui
    except ImportError as exc:
        print(f"[red]TUI import failed:[/red] {exc}")
        print("[yellow]Install textual>=0.60.[/yellow]")
        raise typer.Exit(code=1)
    run_tui(config_path=config, data_root=data_root)


@app.command("viewer")
def viewer_cmd(
    results_dir: Path = typer.Option(
        Path("RESULTS"),
        "--results-dir",
        "-r",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share/--no-share"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    try:
        from .viewer import run_viewer
    except ImportError as exc:
        print(f"[red]Viewer import failed:[/red] {exc}")
        print("[yellow]Install gradio and trimesh.[/yellow]")
        raise typer.Exit(code=1)
    run_viewer(
        results_dir=results_dir,
        host=host,
        port=port,
        share=share,
        inbrowser=not no_browser,
    )


@app.command("calibrate-intrinsics")
def calibrate_intrinsics_cmd(
    image_dir: Path = typer.Option(
        Path("checkerboard"),
        "--image-dir",
        help="Folder containing checkerboard images.",
    ),
    output_dir: Path = typer.Option(
        Path("intrinsics"),
        "--output-dir",
        help="Directory where intrinsics.npy and distortions.npy are saved.",
    ),
    checkerboard_cols: int = typer.Option(
        10,
        "--checkerboard-cols",
        help="Checkerboard inner corners along X (columns).",
    ),
    checkerboard_rows: int = typer.Option(
        7,
        "--checkerboard-rows",
        help="Checkerboard inner corners along Y (rows).",
    ),
    square_size_m: float = typer.Option(
        0.024,
        "--square-size-m",
        help="Physical checkerboard square size in meters.",
    ),
) -> None:
    """Calibrate camera intrinsics from checkerboard images."""
    try:
        mtx, dist = calibrate_intrinsics(
            image_dir=image_dir,
            output_dir=output_dir,
            checkerboard_size=(int(checkerboard_cols), int(checkerboard_rows)),
            square_size_m=float(square_size_m),
        )
    except ValueError as exc:
        print(f"[red]Calibration failed:[/red] {exc}")
        raise typer.Exit(code=1)

    print(f"[green]Saved:[/green] {output_dir / 'intrinsics.npy'}")
    print(f"[green]Saved:[/green] {output_dir / 'distortions.npy'}")
    print("Camera Matrix:")
    print(mtx)
    print("Distortion Coefficients:")
    print(dist)


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        app(prog_name="spectra")
    else:
        app(args=argv, prog_name="spectra")


if __name__ == "__main__":
    main()
