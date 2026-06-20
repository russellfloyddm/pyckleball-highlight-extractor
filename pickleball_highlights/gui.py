"""Gradio GUI for configuring and monitoring highlight extraction runs.

Works in Google Colab and local environments. Pass ``share=True`` to
``main()`` (or run with ``--share``) to expose a public URL when using Colab.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time

from pickleball_highlights.main import PipelineObserver, run

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    GRADIO_IMPORT_ERROR = exc
    gr = None
else:
    GRADIO_IMPORT_ERROR = None


STAGE_LABELS = {
    "startup": "Startup",
    "input": "Input",
    "audio": "Audio",
    "frames": "Frame Processing",
    "scoring": "Scoring",
    "clips": "Clip Generation",
    "compile": "Compilation",
    "done": "Complete",
    "error": "Error",
}


class QueueLogHandler(logging.Handler):
    """Forward log records into a thread-safe queue."""

    def __init__(self, event_queue: "queue.Queue[tuple]") -> None:
        super().__init__()
        self._event_queue = event_queue
        self.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self._event_queue.put(("log", self.format(record)))


class GradioObserver(PipelineObserver):
    """Translate pipeline callbacks into queued events."""

    def __init__(self, event_queue: "queue.Queue[tuple]") -> None:
        self._event_queue = event_queue

    def on_status(self, stage: str, message: str) -> None:
        self._event_queue.put(("status", stage, message))

    def on_progress(self, stage: str, current: int, total: int) -> None:
        self._event_queue.put(("progress", stage, current, total))


def _get_file_path(file_obj: object) -> str:
    """Extract a filesystem path from a Gradio file object or plain string."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name  # type: ignore[union-attr]
    return ""


def _parse_optional_float(value: str, label: str) -> "float | None":
    stripped = str(value).strip() if value is not None else ""
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _run_extraction(
    input_file: object,
    output_folder: str,
    config_file: object,
    threshold: str,
    before: str,
    after: str,
    log_level: str,
    no_audio: bool,
    no_pose: bool,
) -> "object":
    """Run the extraction pipeline and stream status updates via ``yield``.

    Yields tuples of ``(stage, status, progress_pct, log_text)`` so that
    Gradio can update the UI incrementally while the pipeline runs.
    """
    input_path = _get_file_path(input_file)
    config_path = _get_file_path(config_file) or None
    output_dir = str(output_folder or "").strip()

    if not input_path:
        yield "Error", "Input video is required.", 0.0, "[ERROR] Input video is required.\n"
        return
    if not output_dir:
        yield "Error", "Output folder is required.", 0.0, "[ERROR] Output folder is required.\n"
        return

    try:
        threshold_val = _parse_optional_float(threshold, "Threshold")
        before_val = _parse_optional_float(before, "Padding before")
        after_val = _parse_optional_float(after, "Padding after")
    except ValueError as exc:
        yield "Error", str(exc), 0.0, f"[ERROR] {exc}\n"
        return

    args = argparse.Namespace(
        input=input_path,
        output=output_dir,
        config=config_path,
        threshold=threshold_val,
        before=before_val,
        after=after_val,
        no_audio=no_audio,
        no_pose=no_pose,
        log_level=log_level,
        log_file=None,
        start_time=None,
        end_time=None,
    )

    event_queue: "queue.Queue[tuple]" = queue.Queue()
    observer = GradioObserver(event_queue)
    log_handler = QueueLogHandler(event_queue)

    def _worker() -> None:
        try:
            exit_code = run(args, observer=observer, log_handlers=[log_handler])
            event_queue.put(("finished", exit_code))
        except Exception as exc:
            event_queue.put(("exception", str(exc)))
        finally:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    stage_label = "Starting"
    status_msg = "Preparing extraction job."
    progress_pct = 0.0
    log_lines = ""

    yield stage_label, status_msg, progress_pct, log_lines

    done = False
    while not done:
        updated = False
        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "log":
                log_lines += event[1] + "\n"
                updated = True
            elif kind == "status":
                _, stage, message = event
                stage_label = STAGE_LABELS.get(stage, stage.title())
                status_msg = message
                updated = True
            elif kind == "progress":
                _, stage, current, total = event
                if total > 0:
                    progress_pct = min(100.0, (current / total) * 100)
                updated = True
            elif kind == "finished":
                exit_code = event[1]
                status_msg = (
                    "Extraction finished successfully."
                    if exit_code == 0
                    else f"Extraction finished with exit code {exit_code}."
                )
                progress_pct = 100.0
                done = True
                updated = True
            elif kind == "exception":
                stage_label = "Error"
                status_msg = event[1]
                log_lines += f"[ERROR] {event[1]}\n"
                done = True
                updated = True

        if updated or not done:
            yield stage_label, status_msg, progress_pct, log_lines

        if not done:
            time.sleep(0.1)

    yield stage_label, status_msg, progress_pct, log_lines


def create_interface() -> "gr.Blocks":
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Pyckleball Highlight Extractor") as demo:
        gr.Markdown("# 🏓 Pyckleball Highlight Extractor")
        gr.Markdown(
            "Upload a pickleball match video to automatically extract highlight clips. "
            "Works in Google Colab and local environments."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Configuration")
                input_file = gr.File(
                    label="Input video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".m4v"],
                )
                output_folder = gr.Textbox(
                    label="Output folder",
                    placeholder="/content/highlights",
                    info="Path where highlight clips will be saved.",
                )
                config_file = gr.File(
                    label="Config file (optional)",
                    file_types=[".yaml", ".yml"],
                )
                with gr.Row():
                    threshold = gr.Textbox(label="Threshold", placeholder="0.7")
                    before = gr.Textbox(label="Padding before (s)", placeholder="5")
                    after = gr.Textbox(label="Padding after (s)", placeholder="5")
                log_level = gr.Dropdown(
                    label="Log level",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    value="INFO",
                )
                with gr.Row():
                    no_audio = gr.Checkbox(label="Skip audio analysis", value=False)
                    no_pose = gr.Checkbox(label="Skip pose estimation", value=False)
                start_btn = gr.Button("▶ Start Extraction", variant="primary")

            with gr.Column():
                gr.Markdown("### 📊 Status")
                stage_output = gr.Textbox(label="Stage", interactive=False, value="Idle")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Choose an input video and output folder.",
                )
                progress_output = gr.Slider(
                    label="Progress (%)",
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False,
                )
                log_output = gr.Textbox(
                    label="Live Logs",
                    lines=20,
                    max_lines=50,
                    interactive=False,
                    autoscroll=True,
                )

        start_btn.click(
            fn=_run_extraction,
            inputs=[
                input_file,
                output_folder,
                config_file,
                threshold,
                before,
                after,
                log_level,
                no_audio,
                no_pose,
            ],
            outputs=[stage_output, status_output, progress_output, log_output],
        )

    return demo


def main(share: bool = False) -> None:
    """Launch the Gradio GUI.

    Args:
        share: When ``True``, Gradio creates a public shareable URL which is
            required when running inside Google Colab.
    """
    if GRADIO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Gradio is not installed. Install it with: pip install gradio"
        ) from GRADIO_IMPORT_ERROR
    demo = create_interface()
    demo.queue()
    demo.launch(share=share)


if __name__ == "__main__":
    main(share="--share" in sys.argv)
