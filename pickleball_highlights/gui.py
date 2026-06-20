"""Tkinter GUI for configuring and monitoring highlight extraction runs."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
from typing import Callable

from pickleball_highlights.main import PipelineObserver, run

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError as exc:  # pragma: no cover - depends on system package
    TKINTER_IMPORT_ERROR = exc
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
else:
    TKINTER_IMPORT_ERROR = None


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
    """Forward log records into a thread-safe GUI event queue."""

    def __init__(self, event_queue: "queue.Queue[tuple]") -> None:
        super().__init__()
        self._event_queue = event_queue
        self.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self._event_queue.put(("log", self.format(record)))


class GuiObserver(PipelineObserver):
    """Translate pipeline callbacks into queued GUI events."""

    def __init__(self, event_queue: "queue.Queue[tuple]") -> None:
        self._event_queue = event_queue

    def on_status(self, stage: str, message: str) -> None:
        self._event_queue.put(("status", stage, message))

    def on_progress(self, stage: str, current: int, total: int) -> None:
        self._event_queue.put(("progress", stage, current, total))


if tk is not None:
    class HighlightExtractorGUI(tk.Tk):
        """Desktop interface for running and monitoring highlight extraction."""

        def __init__(self) -> None:
            super().__init__()
            self.title("Pyckleball Highlight Extractor")
            self.geometry("900x700")
            self.minsize(780, 620)

            self._event_queue: "queue.Queue[tuple]" = queue.Queue()
            self._running = False

            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()
            self.config_var = tk.StringVar()
            self.threshold_var = tk.StringVar()
            self.before_var = tk.StringVar()
            self.after_var = tk.StringVar()
            self.log_level_var = tk.StringVar(value="INFO")
            self.no_audio_var = tk.BooleanVar(value=False)
            self.no_pose_var = tk.BooleanVar(value=False)
            self.stage_var = tk.StringVar(value="Idle")
            self.status_var = tk.StringVar(value="Choose an input video and output folder.")
            self.progress_var = tk.DoubleVar(value=0.0)
            self.progress_text_var = tk.StringVar(value="No job running.")

            self._build_layout()
            self.after(100, self._process_events)
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        def _build_layout(self) -> None:
            main = ttk.Frame(self, padding=16)
            main.pack(fill=tk.BOTH, expand=True)
            main.columnconfigure(0, weight=1)
            main.rowconfigure(3, weight=1)

            options = ttk.LabelFrame(main, text="Run Configuration", padding=12)
            options.grid(row=0, column=0, sticky="nsew")
            options.columnconfigure(1, weight=1)

            self._add_path_row(
                options,
                row=0,
                label="Input video",
                variable=self.input_var,
                button_text="Browse…",
                command=self._browse_input,
            )
            self._add_path_row(
                options,
                row=1,
                label="Output folder",
                variable=self.output_var,
                button_text="Browse…",
                command=self._browse_output,
            )
            self._add_path_row(
                options,
                row=2,
                label="Config file",
                variable=self.config_var,
                button_text="Browse…",
                command=self._browse_config,
            )

            ttk.Label(options, text="Threshold").grid(row=3, column=0, sticky="w", pady=(10, 0))
            ttk.Entry(options, textvariable=self.threshold_var).grid(
                row=3, column=1, sticky="ew", pady=(10, 0), padx=(0, 8)
            )

            ttk.Label(options, text="Padding before (s)").grid(
                row=4, column=0, sticky="w", pady=(10, 0)
            )
            ttk.Entry(options, textvariable=self.before_var).grid(
                row=4, column=1, sticky="ew", pady=(10, 0), padx=(0, 8)
            )

            ttk.Label(options, text="Padding after (s)").grid(
                row=5, column=0, sticky="w", pady=(10, 0)
            )
            ttk.Entry(options, textvariable=self.after_var).grid(
                row=5, column=1, sticky="ew", pady=(10, 0), padx=(0, 8)
            )

            ttk.Label(options, text="Log level").grid(row=6, column=0, sticky="w", pady=(10, 0))
            ttk.Combobox(
                options,
                textvariable=self.log_level_var,
                values=("DEBUG", "INFO", "WARNING", "ERROR"),
                state="readonly",
            ).grid(row=6, column=1, sticky="ew", pady=(10, 0), padx=(0, 8))

            toggles = ttk.Frame(options)
            toggles.grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 0))
            ttk.Checkbutton(
                toggles, text="Skip audio analysis", variable=self.no_audio_var
            ).pack(side=tk.LEFT, padx=(0, 12))
            ttk.Checkbutton(
                toggles, text="Skip pose estimation", variable=self.no_pose_var
            ).pack(side=tk.LEFT)

            actions = ttk.Frame(main)
            actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
            self.start_button = ttk.Button(actions, text="Start Extraction", command=self._start_run)
            self.start_button.pack(side=tk.LEFT)

            progress = ttk.LabelFrame(main, text="Status", padding=12)
            progress.grid(row=2, column=0, sticky="ew", pady=(12, 0))
            progress.columnconfigure(0, weight=1)

            ttk.Label(progress, textvariable=self.stage_var, font=("", 11, "bold")).grid(
                row=0, column=0, sticky="w"
            )
            ttk.Label(progress, textvariable=self.status_var, wraplength=820).grid(
                row=1, column=0, sticky="w", pady=(4, 8)
            )
            self.progress_bar = ttk.Progressbar(
                progress, variable=self.progress_var, maximum=100, mode="determinate"
            )
            self.progress_bar.grid(row=2, column=0, sticky="ew")
            ttk.Label(progress, textvariable=self.progress_text_var).grid(
                row=3, column=0, sticky="w", pady=(6, 0)
            )

            logs = ttk.LabelFrame(main, text="Live Logs", padding=12)
            logs.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
            logs.columnconfigure(0, weight=1)
            logs.rowconfigure(0, weight=1)

            self.log_text = tk.Text(logs, wrap="word", state="disabled", height=18)
            self.log_text.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.log_text.configure(yscrollcommand=scrollbar.set)

        def _add_path_row(
            self,
            parent: "ttk.LabelFrame",
            row: int,
            label: str,
            variable: "tk.StringVar",
            button_text: str,
            command: Callable[[], None],
        ) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 10))
            ttk.Entry(parent, textvariable=variable).grid(
                row=row, column=1, sticky="ew", padx=(0, 8), pady=(0, 10)
            )
            ttk.Button(parent, text=button_text, command=command).grid(
                row=row, column=2, sticky="ew", pady=(0, 10)
            )

        def _browse_input(self) -> None:
            path = filedialog.askopenfilename(
                title="Select input video",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self.input_var.set(path)

        def _browse_output(self) -> None:
            path = filedialog.askdirectory(title="Select output folder")
            if path:
                self.output_var.set(path)

        def _browse_config(self) -> None:
            path = filedialog.askopenfilename(
                title="Select config file",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            )
            if path:
                self.config_var.set(path)

        def _parse_optional_float(self, value: str, label: str) -> float | None:
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError as exc:
                raise ValueError(f"{label} must be a number.") from exc

        def _build_args(self) -> argparse.Namespace:
            if not self.input_var.get().strip():
                raise ValueError("Input video is required.")
            if not self.output_var.get().strip():
                raise ValueError("Output folder is required.")

            return argparse.Namespace(
                input=self.input_var.get().strip(),
                output=self.output_var.get().strip(),
                config=self.config_var.get().strip() or None,
                threshold=self._parse_optional_float(self.threshold_var.get(), "Threshold"),
                before=self._parse_optional_float(self.before_var.get(), "Padding before"),
                after=self._parse_optional_float(self.after_var.get(), "Padding after"),
                no_audio=self.no_audio_var.get(),
                no_pose=self.no_pose_var.get(),
                log_level=self.log_level_var.get(),
                log_file=None,
            )

        def _start_run(self) -> None:
            if self._running:
                return
            try:
                args = self._build_args()
            except ValueError as exc:
                messagebox.showerror("Invalid configuration", str(exc), parent=self)
                return

            self._running = True
            self.start_button.configure(state="disabled")
            self.stage_var.set("Starting")
            self.status_var.set("Preparing extraction job.")
            self.progress_var.set(0.0)
            self.progress_text_var.set("Waiting for pipeline updates…")
            self._clear_logs()

            worker = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
            worker.start()

        def _run_pipeline(self, args: argparse.Namespace) -> None:
            observer = GuiObserver(self._event_queue)
            log_handler = QueueLogHandler(self._event_queue)
            try:
                exit_code = run(args, observer=observer, log_handlers=[log_handler])
                self._event_queue.put(("finished", exit_code))
            except Exception as exc:
                self._event_queue.put(("exception", str(exc)))
            finally:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()

        def _process_events(self) -> None:
            while True:
                try:
                    event = self._event_queue.get_nowait()
                except queue.Empty:
                    break

                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "status":
                    self._update_status(event[1], event[2])
                elif kind == "progress":
                    self._update_progress(event[1], event[2], event[3])
                elif kind == "finished":
                    self._finish_run(event[1])
                elif kind == "exception":
                    self._fail_run(event[1])

            self.after(100, self._process_events)

        def _update_status(self, stage: str, message: str) -> None:
            self.stage_var.set(STAGE_LABELS.get(stage, stage.title()))
            self.status_var.set(message)
            if stage in {"done", "error"}:
                self.progress_var.set(100.0 if stage == "done" else self.progress_var.get())

        def _update_progress(self, stage: str, current: int, total: int) -> None:
            label = STAGE_LABELS.get(stage, stage.title())
            if total <= 0:
                self.progress_var.set(0.0)
                self.progress_text_var.set(f"{label}: waiting for work")
                return
            percent = (current / total) * 100 if total else 0.0
            self.progress_var.set(max(0.0, min(100.0, percent)))
            self.progress_text_var.set(f"{label}: {current}/{total}")

        def _finish_run(self, exit_code: int) -> None:
            self._running = False
            self.start_button.configure(state="normal")
            if exit_code == 0:
                self.status_var.set("Extraction finished.")
            else:
                self.status_var.set(f"Extraction finished with exit code {exit_code}.")

        def _fail_run(self, message: str) -> None:
            self._running = False
            self.stage_var.set("Error")
            self.status_var.set(message)
            self.start_button.configure(state="normal")
            self._append_log(f"[ERROR] GUI: {message}")
            messagebox.showerror("Extraction failed", message, parent=self)

        def _append_log(self, line: str) -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, f"{line}\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

        def _clear_logs(self) -> None:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.configure(state="disabled")

        def _on_close(self) -> None:
            if self._running:
                should_close = messagebox.askyesno(
                    "Exit",
                    "An extraction is still running. Close the GUI anyway?",
                    parent=self,
                )
                if not should_close:
                    return
            self.destroy()


def main() -> None:
    """Launch the desktop GUI."""
    if TKINTER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tkinter is not installed. Install the Python Tk package for your "
            "platform (for example: python3-tk on Debian/Ubuntu)."
        ) from TKINTER_IMPORT_ERROR
    app = HighlightExtractorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
