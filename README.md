# Pyckleball Highlight Extractor

Automatically extract exciting highlight clips from full pickleball match videos using computer vision, object detection, and audio analysis.

## Features

- 🎾 **Rally Detection** – Detects start/end of rallies via ball tracking and optical flow
- 🏃 **Player Tracking** – Follows players across frames using YOLOv8 + ByteTrack
- 🦾 **Pose Estimation** – Identifies celebrations, lunges, dives, and raised paddles (MediaPipe)
- 🔊 **Audio Analysis** – Detects crowd excitement and volume spikes (librosa)
- 🏆 **Highlight Scoring** – Configurable weighted model combining all signals
- ✂️ **Clip Generation** – Exports padded MP4 clips with FFmpeg
- 🎬 **Highlight Reel** – Concatenates clips with transition effects
- 📊 **Metadata Export** – JSON file with timestamps, scores, and reasons

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) (must be on `PATH`)
- GPU optional (CUDA / Apple Silicon MPS)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/russellfloyddm/pyckleball-highlight-extractor.git
cd pyckleball-highlight-extractor

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### FFmpeg Setup

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**  
Download from https://ffmpeg.org/download.html and add the `bin` directory to your `PATH`.

Verify installation:
```bash
ffmpeg -version
```

## Usage

### Basic

```bash
python -m pickleball_highlights.main \
  --input match.mp4 \
  --output ./highlights
```

### Advanced

```bash
python -m pickleball_highlights.main \
  --input match.mp4 \
  --output ./highlights \
  --config config.yaml \
  --threshold 0.75 \
  --before 5 \
  --after 5 \
  --log-level INFO
```

### All Options

```
usage: pyckleball-highlights [-h] --input VIDEO --output DIR [--config YAML]
                              [--threshold SCORE] [--before SECONDS]
                              [--after SECONDS] [--no-audio] [--no-pose]
                              [--log-level {DEBUG,INFO,WARNING,ERROR}]
                              [--log-file FILE]

options:
  --input, -i VIDEO       Path to the input video file (MP4, MOV, AVI, …)
  --output, -o DIR        Directory where highlight clips and the reel are saved
  --config, -c YAML       Path to a YAML configuration file (optional)
  --threshold SCORE       Minimum highlight score [0–1] (default: 0.7)
  --before SECONDS        Seconds of padding before each highlight (default: 5)
  --after SECONDS         Seconds of padding after each highlight (default: 5)
  --no-audio              Skip audio analysis
  --no-pose               Skip pose estimation
  --log-level             Logging verbosity (default: INFO)
  --log-file FILE         Optional log output file
```

## Output

```
highlights/
├── clips/
│   ├── highlight_001.mp4
│   ├── highlight_002.mp4
│   └── ...
├── highlights_reel.mp4
└── highlights_metadata.json
```

### Metadata Example

```json
{
  "highlights": [
    {
      "index": 1,
      "start": 120.5,
      "end": 145.2,
      "padded_start": 115.5,
      "padded_end": 150.2,
      "duration": 34.7,
      "score": 0.92,
      "reason": "long_rally,fast_exchanges",
      "clip_path": "highlights/clips/highlight_001.mp4"
    }
  ],
  "total_clips": 1
}
```

## Configuration

Copy `config.yaml` and edit as needed:

```yaml
scoring:
  rally_duration: 0.35    # Weight for rally duration
  shot_count: 0.25        # Weight for number of shots
  ball_speed: 0.15        # Weight for ball speed
  movement: 0.15          # Weight for player movement
  audio: 0.10             # Weight for audio excitement

highlight:
  score_threshold: 0.7    # Minimum score to generate a clip
  merge_gap: 10.0         # Merge highlights within this many seconds
  clip_before: 5.0        # Seconds of padding before each highlight
  clip_after: 5.0         # Seconds of padding after each highlight

detection:
  yolo_model: yolov8n.pt  # yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt
  device: auto            # auto / cpu / cuda / mps
```

Pass via `--config`:

```bash
python -m pickleball_highlights.main -i match.mp4 -o ./out --config config.yaml
```

## Highlight Scoring Formula

```
highlight_score =
    0.35 × rally_duration_score   (20 s → 1.0)
  + 0.25 × shot_count_score       (20 shots → 1.0)
  + 0.15 × ball_speed_score       (800 px/s → 1.0)
  + 0.15 × player_movement_score  (300 px/s → 1.0)
  + 0.10 × audio_excitement_score

# Long-rally bonuses (additive, capped at 1.0):
#   10 shots → +0.10
#   20 shots → +0.20
#   30 shots → +0.35
```

All weights are configurable in `config.yaml`.

## Architecture

```
pickleball_highlights/
├── main.py               # CLI entry point
├── config.py             # Configuration dataclasses + YAML loader
├── video_loader.py       # OpenCV video loading & metadata
├── ball_tracker.py       # YOLOv8 ball detection + optical flow fallback
├── player_tracker.py     # YOLOv8 player tracking + optical flow fallback
├── pose_estimator.py     # MediaPipe pose estimation
├── audio_analyzer.py     # librosa / numpy audio excitement analysis
├── rally_detector.py     # Rally detection from ball tracking data
├── highlight_scorer.py   # Weighted highlight scoring model
├── clip_generator.py     # FFmpeg clip extraction
├── highlight_compiler.py # FFmpeg clip concatenation + metadata
├── utils/
│   ├── __init__.py
│   └── logging_utils.py  # Logging setup
└── models/
    └── __init__.py       # Model path resolution
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pickleball_highlights --cov-report=term-missing
```

## Performance Notes

- GPU acceleration is used automatically when available (CUDA / MPS)
- CPU fallback is always available
- Use `yolov8n.pt` for faster processing; `yolov8s.pt` / `yolov8m.pt` for higher accuracy
- `--no-audio` and `--no-pose` flags reduce processing time
- Progress is displayed with `tqdm` progress bars

## Logging

```
[INFO] Pyckleball Highlight Extractor ===
[INFO] Input: match.mp4
[INFO] Detecting audio excitement…
[INFO] Detecting rallies… (54000 frames @ 30.0 fps)
[INFO] Rally detected: 120.50–145.20s (24.7s, 18 shots, avg speed=312.4 px/s)
[INFO] Scoring 12 rally candidates…
[INFO] 8/12 candidates above threshold 0.70
[INFO] Extracting highlight 1/8 (score=0.921, reason=long_rally)
[INFO] Clip saved: highlight_001.mp4 (34.7s)
[INFO] Highlight reel: highlights/highlights_reel.mp4
[INFO] Metadata: highlights/highlights_metadata.json
[INFO] Done. Generated 8 highlight clip(s).
```