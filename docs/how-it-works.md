# How the Current System Works

This document describes the code that currently exists in the repository. It focuses on the real runtime flow in `pickleball_highlights.main.run`, the supporting modules it calls, and a few implementation details that are easy to miss from the high-level README alone.

## Entry points

The project has two user-facing entry points:

- **CLI:** `python -m pickleball_highlights.main`
- **Desktop GUI:** `python -m pickleball_highlights.gui`

The GUI is a thin wrapper around the same pipeline used by the CLI. It builds an `argparse.Namespace`, calls `run(...)`, and listens for status/progress callbacks so it can update the window and live log pane.

## End-to-end pipeline

The main pipeline lives in `pickleball_highlights/main.py` and runs in this order:

1. **Configure logging**
   - `setup_logging(...)` configures stdout logging and optionally file logging.
   - The GUI can also attach an extra log handler so logs are streamed into the interface.

2. **Load configuration**
   - `load_config(...)` reads YAML into the `AppConfig` dataclasses in `config.py`.
   - If no config file is provided, the code uses the built-in defaults.
   - CLI flags can override the loaded highlight threshold and clip padding values.

3. **Read video metadata**
   - `VideoLoader` opens the input file with OpenCV and reads FPS, resolution, frame count, duration, and codec.
   - The loader also exposes a frame iterator that yields `(frame_index, timestamp, frame)` tuples.

4. **Analyze audio before frame processing**
   - `AudioAnalyzer.analyze(...)` extracts the audio track to a temporary WAV with FFmpeg.
   - It then computes per-window excitement scores across the full file.
   - If `librosa` is available, the analyzer uses that path.
   - If `librosa` is missing, it falls back to a simpler NumPy RMS-based analysis.
   - If the user passes `--no-audio`, this stage is skipped and audio contributes `0` later.

5. **Initialize vision components**
   - `BallTracker` handles ball detection/tracking.
   - `PlayerTracker` handles player detection/tracking and movement scoring.
   - `PoseEstimator` handles pose classification unless `--no-pose` is used.
   - `RallyDetector` turns ball visibility and speed history into rally segments.

6. **Process the video frame by frame**
   - The pipeline iterates through every frame from `VideoLoader.frames()`.
   - For each frame it:
     - runs the ball tracker
     - runs the player tracker
     - derives a movement score from player tracks
     - runs pose estimation every fifth frame
     - slightly boosts movement when pose detection reports `CELEBRATION` or `RAISED_PADDLE`
     - updates the rally detector
   - Progress updates are emitted periodically for the CLI/GUI observer.

7. **Flush the last rally**
   - After the frame loop, `RallyDetector.flush(...)` closes any in-progress rally that reached the end of the video.

8. **Score rally candidates**
   - `HighlightScorer` converts each detected rally into a `HighlightCandidate`.
   - Scores combine rally duration, shot count, ball speed, movement, and audio excitement.
   - Long rallies receive an additive bonus before the score is capped at `1.0`.

9. **Generate clips**
   - `ClipGenerator.generate(...)` filters out low-scoring candidates.
   - Nearby highlights are merged if the gap between them is below `highlight.merge_gap`.
   - Each merged highlight is padded with `clip_before` and `clip_after`.
   - FFmpeg extracts each clip into `output/clips/highlight_XXX.mp4`.

10. **Compile the reel and write metadata**
    - `HighlightCompiler.compile(...)` concatenates generated clips into `highlights_reel.mp4`.
    - `HighlightCompiler.write_metadata(...)` always writes `highlights_metadata.json` for the clips that were successfully generated.

11. **Return success**
    - The pipeline logs the final clip count, sends a final observer status update, and returns exit code `0`.

## Component behavior

### `config.py`

`config.py` defines the application settings as dataclasses:

- `ScoringWeights`
- `DetectionConfig`
- `AudioConfig`
- `RallyConfig`
- `HighlightConfig`
- `AppConfig`

Notable details:

- scoring weights must sum to `1.0`
- YAML rally bonus keys are converted back to integers after parsing
- missing config files are treated as "use defaults"

### `video_loader.py`

`VideoLoader` is responsible for:

- validating that the file exists
- validating the extension
- reading metadata lazily
- iterating frames with timestamps
- reading a single frame at a specific timestamp

Although `frames(...)` accepts `batch_size`, the current implementation still yields one frame at a time.

### `ball_tracker.py`

`BallTracker` tries to load an Ultralytics YOLO model and predict only the configured ball class.

If that model is unavailable, it falls back to a motion-based approach:

- convert frame to grayscale
- compare against the previous frame
- threshold the difference image
- take the largest contour as the moving ball proxy

The tracker keeps a `BallTrackState` with:

- raw detections
- per-step speed estimates in pixels per second

### `player_tracker.py`

`PlayerTracker` also prefers Ultralytics YOLO, using `track(...)` with ByteTrack persistence.

If YOLO is unavailable, it falls back to dense optical flow:

- compute Farneback optical flow between consecutive grayscale frames
- threshold motion magnitude
- keep the largest motion regions as synthetic player detections

The module stores per-player paths over time and exposes a normalized combined movement score for the current run.

### `pose_estimator.py`

`PoseEstimator` prefers MediaPipe Pose and falls back to lightweight frame heuristics if MediaPipe is unavailable.

The pipeline currently uses pose results in a limited way:

- pose estimation runs every fifth frame
- only `CELEBRATION` and `RAISED_PADDLE` currently influence scoring
- `LUNGE` and `DIVE` are classified by the estimator but are not directly used by `main.py`

### `rally_detector.py`

`RallyDetector` treats ball visibility as the main signal for rally segmentation.

- a rally starts when the ball first appears
- a rally ends when ball absence exceeds `max_gap_duration`
- short rallies and rallies with too few inferred shots are discarded

Shot counting is based on a speed-peak heuristic over recent ball speeds, not on explicit hit classification.

Each completed `Rally` stores:

- start and end times
- shot count
- max and average ball speed
- a smoothed movement score

### `highlight_scorer.py`

`HighlightScorer` calculates the final score for each rally from:

- normalized rally duration
- normalized shot count
- normalized average ball speed
- movement score
- average audio excitement during the rally

It also builds a simple comma-separated reason string such as:

- `very_long_rally`
- `long_rally`
- `medium_rally`
- `fast_exchanges`
- `extended_point`

### `clip_generator.py`

`ClipGenerator` is responsible for the per-highlight video exports.

Current behavior:

- filter by threshold
- merge nearby candidates
- clamp padded clip boundaries to the video duration
- call FFmpeg once per generated clip

The clip metadata is stored in `GeneratedClip`.

### `highlight_compiler.py`

`HighlightCompiler` creates the final reel and metadata file.

Current behavior:

- writes a temporary FFmpeg concat list
- concatenates clips into one output reel
- writes JSON metadata with timestamps, padded bounds, scores, reasons, and paths

Important implementation detail:

- `add_transitions` currently changes whether the final reel is re-encoded or stream-copied
- the code does **not** currently build actual cross-fade transitions
- `transition_duration`, `title_screen`, `title_text`, and `title_duration` are loaded from config but are not currently used by the compiler

## GUI integration

The GUI in `gui.py` does not implement a separate pipeline. Instead, it layers a user interface on top of `run(...)`.

It does this by:

- collecting file paths and option values from Tkinter widgets
- running the pipeline on a background thread
- receiving pipeline stage updates through a `PipelineObserver`
- receiving log lines through a custom logging handler

The stage names currently used by the pipeline are:

- `startup`
- `input`
- `audio`
- `frames`
- `scoring`
- `clips`
- `compile`
- `done`
- `error`

## Current limitations and implementation notes

Some configuration or architecture hooks exist but are not fully used yet:

- `batch_size` is loaded into `AppConfig` but is not used in the frame-processing loop
- `max_workers` is loaded into `AppConfig` but the pipeline currently runs in a single main processing loop
- `VideoLoader.frames(batch_size=...)` accepts a batch size but still yields one frame at a time
- pose events beyond celebration and raised paddle are not yet wired into scoring
- reel "transitions" are currently re-encoding behavior rather than true transition effects

These are useful to know when comparing the README's high-level feature list with the exact behavior of the current codebase.
