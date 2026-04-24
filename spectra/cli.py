"""Typer CLI for the ArUco-stabilized reconstruction pipeline.

After ``pip install -e . --no-deps`` from the repo root, invoke as ``spectra``.
Otherwise fall back to ``python -m spectra`` (both are equivalent).

Usage::

    # Run with the default config
    spectra reconstruct --config configs/default.yaml

    # Override config fields on the command line
    spectra reconstruct \
        --config configs/default.yaml \
        --rgb-dir DATA/SAMPLE1_iphone/rgb \
        --backend mast3r \
        --set aruco.marker_edge_length_m=0.03

    # Detect markers in a folder (legacy `detect_aruco.py` replacement)
    spectra detect-aruco DATA/SAMPLE1_iphone/rgb /tmp/aruco_out

    # Launch the Textual TUI
    spectra tui

    # Launch the local Gradio 3D viewer
    spectra viewer --results-dir RESULTS --port 7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from .aruco import ARUCO_DICTIONARIES, detect_folder
from .config import (
    ArucoConfig,
    InputConfig,
    OutputConfig,
    ReconstructionConfig,
    SurfaceConfig,
    VggtConfig,
    Mast3rConfig,
    RerunConfig,
    load_config,
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ArUco-stabilized 3D reconstruction pipeline (VGGT | MASt3R).",
)


def _parse_override(value: str) -> tuple[str, object]:
    """Parse ``key.path=value`` into ``(dotted_key, parsed_value)``.

    The value is parsed with ``json.loads`` when possible; otherwise left as a
    string so the user can pass scalars, booleans, or full JSON literals.
    """
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


def _build_config_from_cli(
    config_path: Optional[Path],
    rgb_dir: Optional[Path],
    pose_dir: Optional[Path],
    camera_params_dir: Optional[Path],
    out_dir: Optional[Path],
    run_name: Optional[str],
    backend: Optional[str],
    marker_edge_length_m: Optional[float],
    aruco_dictionary: Optional[str],
    align_to_aruco: Optional[bool],
    origin_marker_id: Optional[int],
    grpc_port: Optional[int],
    no_wait: bool,
    overrides: List[str],
) -> ReconstructionConfig:
    if config_path is not None:
        cfg = load_config(config_path)
    else:
        if rgb_dir is None:
            raise typer.BadParameter(
                "Either --config or --rgb-dir must be provided."
            )
        cfg = ReconstructionConfig(
            input=InputConfig(rgb_dir=rgb_dir, pose_dir=pose_dir, camera_params_dir=camera_params_dir),
            output=OutputConfig(),
            backend="vggt",
            aruco=ArucoConfig(),
            surface=SurfaceConfig(),
            vggt=VggtConfig(),
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
    if backend is not None:
        direct["backend"] = backend
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


@app.command("reconstruct")
def reconstruct_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="Path to a YAML configuration file.",
    ),
    rgb_dir: Optional[Path] = typer.Option(None, "--rgb-dir", help="Folder with input images."),
    pose_dir: Optional[Path] = typer.Option(None, "--pose-dir", help="Folder with GT poses (optional)."),
    camera_params_dir: Optional[Path] = typer.Option(
        None, "--camera-params-dir", help="Folder with intrinsics.npy (optional)."
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Root folder for run outputs."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional subfolder name for this run."),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        case_sensitive=False,
        help="vggt | mast3r",
    ),
    marker_edge_length_m: Optional[float] = typer.Option(
        None, "--marker-edge-length-m", help="Physical edge length of one ArUco marker (meters)."
    ),
    aruco_dictionary: Optional[str] = typer.Option(
        None, "--aruco-dictionary", help="OpenCV ArUco dictionary name (e.g. 4x4_50)."
    ),
    align_to_aruco: Optional[bool] = typer.Option(
        None, "--align-to-aruco/--no-align-to-aruco", help="Enable/disable ArUco-based Sim3 alignment."
    ),
    origin_marker_id: Optional[int] = typer.Option(
        None, "--origin-marker-id", help="If set, this marker defines the output-frame origin."
    ),
    grpc_port: Optional[int] = typer.Option(None, "--grpc-port", help="Rerun gRPC server port."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Do not block on user input after logging."),
    set_override: List[str] = typer.Option(
        [],
        "--set",
        "-s",
        help="Override any config field, dotted-path: --set aruco.marker_edge_length_m=0.03",
    ),
) -> None:
    """Run the unified reconstruction pipeline."""
    cfg = _build_config_from_cli(
        config_path=config,
        rgb_dir=rgb_dir,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        out_dir=out_dir,
        run_name=run_name,
        backend=backend.lower() if backend else None,
        marker_edge_length_m=marker_edge_length_m,
        aruco_dictionary=aruco_dictionary,
        align_to_aruco=align_to_aruco,
        origin_marker_id=origin_marker_id,
        grpc_port=grpc_port,
        no_wait=no_wait,
        overrides=set_override,
    )

    from .pipeline import run_reconstruction

    result = run_reconstruction(cfg)
    print(f"[green]Run directory:[/green] {result.run_dir}")


@app.command("detect-aruco")
def detect_aruco_cmd(
    input_folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder with input images."),
    output_folder: Path = typer.Argument(..., help="Folder that will receive json/ and annotated/ subdirs."),
    dictionary: str = typer.Option(
        "4x4_50",
        "--dict",
        help=f"OpenCV ArUco dictionary name. One of: {sorted(ARUCO_DICTIONARIES.keys())}",
    ),
    draw_scale: float = typer.Option(
        2.0, "--draw-scale", help="Annotation thickness/text size multiplier (default: 2.0)."
    ),
) -> None:
    """Detect ArUco markers in a folder of images (legacy `detect_aruco.py` replacement)."""
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
    print(f"JSON outputs: [green]{output_folder / 'json'}[/green]")
    print(f"Annotated images: [green]{output_folder / 'annotated'}[/green]")


@app.command("tui")
def tui_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional YAML config to pre-load into the editor.",
    ),
    data_root: Path = typer.Option(
        Path("DATA"),
        "--data-root",
        help="Root folder browsed for sample directories.",
    ),
) -> None:
    """Launch the Textual TUI (sample browser, config editor, live log)."""
    try:
        from .tui import run_tui
    except ImportError as exc:
        print(f"[red]Failed to import TUI module:[/red] {exc}")
        print("[yellow]Install `textual>=0.60` to use the TUI.[/yellow]")
        raise typer.Exit(code=1)
    run_tui(config_path=config, data_root=data_root)


@app.command("viewer")
def viewer_cmd(
    results_dir: Path = typer.Option(
        Path("RESULTS"),
        "--results-dir",
        "-r",
        help="Root folder browsed for run outputs.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (default: localhost)."),
    port: int = typer.Option(7860, "--port", help="HTTP port."),
    share: bool = typer.Option(False, "--share/--no-share", help="Open a public Gradio tunnel."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not auto-open a browser tab."
    ),
) -> None:
    """Launch the local Gradio 3D viewer (cloud + ArUcos + optional surface)."""
    try:
        from .viewer import run_viewer
    except ImportError as exc:
        print(f"[red]Failed to import viewer module:[/red] {exc}")
        print("[yellow]Install `gradio` and `trimesh` to use the viewer.[/yellow]")
        raise typer.Exit(code=1)
    run_viewer(
        results_dir=results_dir,
        host=host,
        port=port,
        share=share,
        inbrowser=not no_browser,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Console-script entry point used by ``spectra`` and ``python -m spectra``."""
    if argv is None:
        app(prog_name="spectra")
    else:
        app(args=argv, prog_name="spectra")


if __name__ == "__main__":
    main()
