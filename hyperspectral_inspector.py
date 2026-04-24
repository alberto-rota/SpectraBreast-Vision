#!/usr/bin/env python3
"""Interactive ENVI hyperspectral inspector with Textual + Rerun."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer


app = typer.Typer(add_completion=False, help="Inspect ENVI hyperspectral cubes in a Textual TUI.")


@dataclass(frozen=True)
class EnviDataset:
    """Resolved ENVI dataset paths."""

    name: str
    hdr_path: Path
    data_path: Path | None


def _discover_hdr_files(inputs: list[Path], recursive: bool) -> list[Path]:
    hdr_files: list[Path] = []
    for in_path in inputs:
        if in_path.is_file():
            if in_path.suffix.lower() == ".hdr":
                hdr_files.append(in_path.resolve())
        elif in_path.is_dir():
            pattern = "**/*.hdr" if recursive else "*.hdr"
            hdr_files.extend(p.resolve() for p in in_path.glob(pattern) if p.is_file())
    unique_sorted = sorted(set(hdr_files))
    return unique_sorted


def _parse_data_file_from_hdr(hdr_path: Path) -> Path | None:
    """Try to parse ENVI `data file = ...` from header."""
    try:
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        line_l = line.strip().lower()
        if line_l.startswith("data file"):
            parts = line.split("=", maxsplit=1)
            if len(parts) != 2:
                continue
            rel = parts[1].strip().strip('"').strip("'")
            if not rel:
                continue
            candidate = (hdr_path.parent / rel).resolve()
            if candidate.exists():
                return candidate
            return candidate
    return None


def discover_datasets(inputs: list[Path], recursive: bool) -> list[EnviDataset]:
    hdr_files = _discover_hdr_files(inputs=inputs, recursive=recursive)
    datasets = [
        EnviDataset(name=hdr.stem, hdr_path=hdr, data_path=_parse_data_file_from_hdr(hdr))
        for hdr in hdr_files
    ]
    return datasets


def load_cube(dataset: EnviDataset) -> np.ndarray:
    """Load ENVI cube as float32 array shaped [H, W, B]."""
    try:
        from spectral import io as spectral_io  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `spectral`. Install with `pip install spectral`."
        ) from exc

    if dataset.data_path is not None:
        spy = spectral_io.envi.open(str(dataset.hdr_path), str(dataset.data_path))
    else:
        spy = spectral_io.envi.open(str(dataset.hdr_path))

    cube = np.asarray(spy.load(), dtype=np.float32)  # [H, W, B]
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube [H, W, B], got shape {cube.shape} for {dataset.name}")
    return cube


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a single band to uint8 for display."""
    finite_mask = np.isfinite(band)
    if not finite_mask.any():
        return np.zeros_like(band, dtype=np.uint8)
    vmin = float(np.nanpercentile(band[finite_mask], 1.0))
    vmax = float(np.nanpercentile(band[finite_mask], 99.0))
    if vmax <= vmin:
        return np.zeros_like(band, dtype=np.uint8)
    norm = (band - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    out = (norm * 255.0).astype(np.uint8)
    return out


def _safe_wavelengths(hdr_path: Path, band_count: int) -> np.ndarray:
    """Read wavelengths from header when available."""
    try:
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return np.arange(band_count, dtype=np.float32)

    lower = text.lower()
    token = "wavelength = {"
    start = lower.find(token)
    if start < 0:
        return np.arange(band_count, dtype=np.float32)
    end = lower.find("}", start)
    if end < 0:
        return np.arange(band_count, dtype=np.float32)
    raw = text[start + len(token) : end]
    vals: list[float] = []
    for x in raw.replace("\n", " ").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:
            continue
    if len(vals) != band_count:
        return np.arange(band_count, dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)


def _cube_summary(cube: np.ndarray) -> str:
    h, w, b = cube.shape
    return (
        f"shape=[H={h}, W={w}, B={b}] "
        f"dtype={cube.dtype} min={np.nanmin(cube):.3f} max={np.nanmax(cube):.3f}"
    )


def _rerun_flush(rr_module: object) -> None:
    """Push batched Rerun data to the gRPC sink."""
    try:
        flush_fn = getattr(rr_module, "flush", None)
        if callable(flush_fn):
            flush_fn()
            return
        for getter_name in ("get_data_recording", "get_global_data_recording"):
            getter = getattr(rr_module, getter_name, None)
            if not callable(getter):
                continue
            rec = getter()
            if rec is not None and hasattr(rec, "flush"):
                rec.flush()
                return
    except Exception:
        return


def run_textual_viewer(
    datasets: list[EnviDataset],
    recording_name: str,
    grpc_port: int,
) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `rerun-sdk`. Install with `pip install rerun-sdk`."
        ) from exc

    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Footer, Header, Static
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `textual`. Install with `pip install textual`."
        ) from exc

    try:
        rr.disconnect()
    except Exception:
        pass
    rr.init(recording_name)
    rr.serve_grpc(grpc_port=grpc_port)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Current band",
                    origin="/",
                    contents=["/hyperspectral/current_band"],
                ),
                rrb.TimeSeriesView(
                    name="Mean spectrum",
                    origin="/",
                    contents=["/hyperspectral/mean_spectrum"],
                ),
            )
        ),
        make_active=True,
    )
    _rerun_flush(rr)

    class HyperspectralApp(App[None]):
        CSS = """
        Screen { layout: vertical; }
        #main { height: 1fr; }
        #left_panel { width: 30%; border: solid $accent; }
        #right_panel { width: 70%; border: solid $accent; padding: 1 2; }
        #status { height: auto; }
        #help { height: auto; color: $text-muted; }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("j,down", "next_band", "Next Band"),
            Binding("k,up", "prev_band", "Prev Band"),
            Binding("n", "next_file", "Next Cube"),
            Binding("p", "prev_file", "Prev Cube"),
            Binding("r", "reload", "Reload Cube"),
        ]

        def __init__(self, ds: list[EnviDataset]) -> None:
            super().__init__()
            self.datasets = ds
            self.dataset_idx = 0
            self.band_idx = 0
            self.cube: np.ndarray | None = None
            self.wavelengths: np.ndarray | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="left_panel"):
                    yield DataTable(id="band_table")
                with Vertical(id="right_panel"):
                    yield Static("", id="status")
                    yield Static(
                        "Keys: j/k or arrows -> band | n/p -> cube | r -> reload | q -> quit",
                        id="help",
                    )
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#band_table", DataTable)
            table.add_columns("Band", "Wavelength", "Mean", "Std")
            self._load_current_dataset()

        def _load_current_dataset(self) -> None:
            dataset = self.datasets[self.dataset_idx]
            self.cube = load_cube(dataset)
            self.band_idx = min(self.band_idx, self.cube.shape[2] - 1)
            self.wavelengths = _safe_wavelengths(dataset.hdr_path, self.cube.shape[2])
            self._refresh_table()
            self._publish_current_band()

        def _refresh_table(self) -> None:
            assert self.cube is not None
            assert self.wavelengths is not None
            table = self.query_one("#band_table", DataTable)
            table.clear(columns=False)

            means = self.cube.mean(axis=(0, 1), dtype=np.float64)  # [B]
            stds = self.cube.std(axis=(0, 1), dtype=np.float64)  # [B]

            band_count = self.cube.shape[2]
            for bi in range(band_count):
                table.add_row(
                    str(bi),
                    f"{self.wavelengths[bi]:.3f}",
                    f"{means[bi]:.4f}",
                    f"{stds[bi]:.4f}",
                    key=f"band-{bi}",
                )
            table.cursor_type = "row"
            table.move_cursor(row=self.band_idx, animate=False)

        def _publish_current_band(self) -> None:
            assert self.cube is not None
            assert self.wavelengths is not None
            dataset = self.datasets[self.dataset_idx]
            band = self.cube[:, :, self.band_idx]  # [H, W]
            band_u8 = normalize_band(band)  # [H, W], uint8

            rr.set_time_sequence("band_index", self.band_idx)
            rr.log("hyperspectral/current_band", rr.Image(band_u8))

            spectral_mean = self.cube.mean(axis=(0, 1), dtype=np.float64)  # [B]
            rr.log(
                "hyperspectral/mean_spectrum",
                rr.SeriesLine(name=f"{dataset.name}_mean", color=[70, 160, 255]),
                static=True,
            )
            rr.send_columns(
                "hyperspectral/mean_spectrum",
                indexes=[rr.TimeColumn("band_index", sequence=np.arange(self.cube.shape[2], dtype=np.int64))],
                columns=[*rr.Scalars.columns(scalars=spectral_mean.astype(np.float32))],
            )
            _rerun_flush(rr)

            status = self.query_one("#status", Static)
            status.update(
                "\n".join(
                    [
                        f"Dataset: {dataset.name}",
                        f"HDR: {dataset.hdr_path}",
                        f"Data: {dataset.data_path if dataset.data_path else '<auto>'}",
                        _cube_summary(self.cube),
                        f"Current band: {self.band_idx}  wavelength={self.wavelengths[self.band_idx]:.3f}",
                        "Rerun entities: hyperspectral/current_band, hyperspectral/mean_spectrum",
                    ]
                )
            )

        def action_next_band(self) -> None:
            assert self.cube is not None
            self.band_idx = min(self.band_idx + 1, self.cube.shape[2] - 1)
            self.query_one("#band_table", DataTable).move_cursor(row=self.band_idx, animate=False)
            self._publish_current_band()

        def action_prev_band(self) -> None:
            assert self.cube is not None
            self.band_idx = max(self.band_idx - 1, 0)
            self.query_one("#band_table", DataTable).move_cursor(row=self.band_idx, animate=False)
            self._publish_current_band()

        def action_next_file(self) -> None:
            self.dataset_idx = (self.dataset_idx + 1) % len(self.datasets)
            self.band_idx = 0
            self._load_current_dataset()

        def action_prev_file(self) -> None:
            self.dataset_idx = (self.dataset_idx - 1) % len(self.datasets)
            self.band_idx = 0
            self._load_current_dataset()

        def action_reload(self) -> None:
            self._load_current_dataset()

    HyperspectralApp(datasets).run()


@app.command("run")
def cli_run(
    inputs: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Input .hdr files and/or folders containing ENVI datasets.",
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Recursively scan folders for .hdr files."
    ),
    recording_name: str = typer.Option(
        "hyperspectral_inspector", "--recording-name", help="Rerun recording name."
    ),
    grpc_port: int = typer.Option(9876, "--grpc-port", help="Rerun gRPC port."),
) -> None:
    """
    Run interactive hyperspectral inspector.
    """
    datasets = discover_datasets(inputs=inputs, recursive=recursive)
    if not datasets:
        typer.echo("No ENVI headers (.hdr) found in provided input paths.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(datasets)} dataset(s). Launching TUI + Rerun...")
    run_textual_viewer(
        datasets=datasets,
        recording_name=recording_name,
        grpc_port=grpc_port,
    )


if __name__ == "__main__":
    app()
