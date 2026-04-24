"""Textual TUI for the ArUco-stabilized reconstruction pipeline.

Three-pane layout:

- **Left**: a directory tree rooted at ``DATA/`` so the user can pick a
  sample folder (the tree is pre-expanded and lets the user click any folder
  containing an ``rgb/`` subdir).
- **Center**: an editable YAML config, with ``Validate`` and ``Run`` buttons.
- **Right**: live stdout/stderr of the running pipeline and a status bar.

The pipeline itself runs in a background thread (Textual ``worker``); the
main event loop stays responsive and the log pane updates as lines arrive.
"""

from __future__ import annotations

import io
import queue
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    RichLog,
    Static,
    TextArea,
)

from .config import ReconstructionConfig, load_config


DEFAULT_YAML = """\
input:
  rgb_dir: DATA/SAMPLE1_iphone/rgb
  pose_dir: null
  camera_params_dir: null

output:
  root: RESULTS
  run_name: null

backend: vggt

aruco:
  dictionary: 4x4_50
  marker_edge_length_m: 0.025
  origin_marker_id: null
  min_views_per_marker: 2
  align_to_aruco: true
  max_sim3_scale_deviation_when_poses_known: 0.05

surface:
  grid_step: 0.0
  fill_iters: 2
  smooth_iters: 1
  min_neighbors: 3
  max_resolution: 2048

vggt:
  model_name: facebook/VGGT-1B
  image_size: 518
  conf_thres: 50.0
  cloud_source: depth_map
  camera_source: predicted
  alignment_mode: sim3
  mask_black_bg: false
  mask_white_bg: false

mast3r:
  model_name: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
  image_size: 512
  neighbor_window: 2
  desc_conf_thr: 0.1
  dense_conf_thr: 12.0
  voxel_size: 0.0015
  max_points: 2000000

rerun:
  enabled: true
  grpc_port: 9876
  no_wait: true
"""


class _LogLine(Message):
    """A line of text produced by the running pipeline."""

    def __init__(self, line: str) -> None:
        self.line = line
        super().__init__()


class _RunFinished(Message):
    """Pipeline run finished (success or failure)."""

    def __init__(self, success: bool, summary: str) -> None:
        self.success = success
        self.summary = summary
        super().__init__()


class _QueueStream(io.TextIOBase):
    """File-like object that forwards writes to a ``queue.Queue``."""

    def __init__(self, out_queue: "queue.Queue[str]") -> None:
        self._queue = out_queue
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:  # type: ignore[override]
        self._buffer += text
        while "\n" in self._buffer:
            line, _, rest = self._buffer.partition("\n")
            self._queue.put(line)
            self._buffer = rest
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._queue.put(self._buffer)
            self._buffer = ""


class ReconstructionTUI(App[None]):
    """Textual app that drives `spectra.pipeline.run_reconstruction`."""

    CSS = """
    #left  { width: 30%; border: round $accent; }
    #mid   { width: 40%; border: round $accent; }
    #right { width: 30%; border: round $accent; }
    #yaml  { height: 1fr; }
    #log   { height: 1fr; }
    #status { height: 3; padding: 0 1; color: $text-muted; }
    .title { background: $boost; color: $text; padding: 0 1; }
    Button { margin: 0 1; }
    """

    BINDINGS = [
        Binding("ctrl+r", "run", "Run reconstruction"),
        Binding("ctrl+v", "validate", "Validate config"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    status_text: reactive[str] = reactive("Ready.", init=False)

    def __init__(
        self,
        config_path: Optional[Path] = None,
        data_root: Path = Path("DATA"),
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).resolve() if Path(data_root).exists() else Path.cwd()
        self._initial_config_path = config_path
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Static("Samples", classes="title")
                yield DirectoryTree(str(self.data_root), id="tree")
            with Vertical(id="mid"):
                yield Static("Config (YAML)", classes="title")
                yield TextArea(self._initial_yaml(), language="yaml", id="yaml")
                with Horizontal():
                    yield Button("Validate", id="validate", variant="primary")
                    yield Button("Run", id="run", variant="success")
            with Vertical(id="right"):
                yield Static("Log", classes="title")
                yield RichLog(id="log", wrap=False, highlight=False, markup=True)
                yield Static(self.status_text, id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._poll_timer = self.set_interval(0.1, self._drain_queue)

    def _initial_yaml(self) -> str:
        if self._initial_config_path is not None and self._initial_config_path.exists():
            try:
                return self._initial_config_path.read_text(encoding="utf-8")
            except OSError:
                pass
        return DEFAULT_YAML

    def watch_status_text(self, status: str) -> None:
        try:
            status_widget = self.query_one("#status", Static)
        except NoMatches:
            return
        status_widget.update(status)

    def _current_yaml(self) -> str:
        return self.query_one("#yaml", TextArea).text

    def _parse_config(self) -> Optional[ReconstructionConfig]:
        try:
            data = yaml.safe_load(self._current_yaml()) or {}
            return ReconstructionConfig.model_validate(data)
        except (yaml.YAMLError, ValidationError) as exc:
            self.status_text = f"Config invalid: {exc}"
            self._log(f"[red]Config invalid:[/red] {exc}")
            return None

    def _log(self, line: str) -> None:
        self.query_one("#log", RichLog).write(line)

    def _drain_queue(self) -> None:
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            self._log(line)

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """When the user picks a folder containing `rgb/`, inject it into the config."""
        path = event.path
        rgb_candidate = Path(path) / "rgb"
        if not rgb_candidate.is_dir():
            return
        try:
            data = yaml.safe_load(self._current_yaml()) or {}
        except yaml.YAMLError:
            data = {}
        data.setdefault("input", {})["rgb_dir"] = str(rgb_candidate)
        pose_candidate = Path(path) / "poses"
        camera_params_candidate = Path(path) / "camera_parameters"
        data["input"]["pose_dir"] = str(pose_candidate) if pose_candidate.is_dir() else None
        data["input"]["camera_params_dir"] = (
            str(camera_params_candidate) if camera_params_candidate.is_dir() else None
        )
        new_text = yaml.safe_dump(data, sort_keys=False)
        self.query_one("#yaml", TextArea).load_text(new_text)
        self.status_text = f"Selected sample: {rgb_candidate}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "validate":
            self.action_validate()
        elif event.button.id == "run":
            self.action_run()

    def action_validate(self) -> None:
        cfg = self._parse_config()
        if cfg is not None:
            self.status_text = f"Config OK (backend={cfg.backend})."
            self._log(f"[green]Config OK:[/green] backend={cfg.backend}, rgb_dir={cfg.input.rgb_dir}")

    def action_run(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.status_text = "A reconstruction is already running."
            return
        cfg = self._parse_config()
        if cfg is None:
            return
        self.status_text = f"Running {cfg.backend} reconstruction on {cfg.input.rgb_dir}..."
        self._log(f"[bold cyan]Starting run:[/bold cyan] backend={cfg.backend}")
        self._worker_thread = threading.Thread(
            target=self._run_worker, args=(cfg,), daemon=True
        )
        self._worker_thread.start()

    def _run_worker(self, cfg: ReconstructionConfig) -> None:
        from .pipeline import run_reconstruction

        stream = _QueueStream(self._queue)
        t0 = time.time()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = run_reconstruction(cfg)
            summary = f"Run {result.run_dir} finished in {time.time() - t0:.1f} s."
            self.call_from_thread(self._on_finished, True, summary)
        except Exception as exc:  # pragma: no cover - runtime errors reported to user
            self.call_from_thread(self._on_finished, False, f"Failed: {exc!r}")

    def _on_finished(self, success: bool, summary: str) -> None:
        if success:
            self.status_text = summary
            self._log(f"[green]{summary}[/green]")
        else:
            self.status_text = summary
            self._log(f"[red]{summary}[/red]")


def run_tui(config_path: Optional[Path] = None, data_root: Path = Path("DATA")) -> None:
    """Launch the Textual TUI from the CLI."""
    app = ReconstructionTUI(config_path=config_path, data_root=data_root)
    app.run()


__all__ = ["ReconstructionTUI", "run_tui"]
