"""Main entry point for the Pyckleball Highlight Extractor CLI."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from tqdm import tqdm

from pickleball_highlights.audio_analyzer import AudioAnalyzer
from pickleball_highlights.ball_tracker import BallTracker
from pickleball_highlights.clip_generator import ClipGenerator
from pickleball_highlights.config import AppConfig, load_config
from pickleball_highlights.highlight_compiler import HighlightCompiler
from pickleball_highlights.highlight_scorer import HighlightScorer
from pickleball_highlights.player_tracker import PlayerTracker
from pickleball_highlights.pose_estimator import PoseEstimator, PoseEvent
from pickleball_highlights.rally_detector import RallyDetector
from pickleball_highlights.utils import get_logger, setup_logging
from pickleball_highlights.video_loader import VideoLoader

logger = get_logger(__name__)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        prog="pyckleball-highlights",
        description="Automatically extract highlight clips from a pickleball match video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="VIDEO",
        help="Path to the input video file (MP4, MOV, AVI, …).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="DIR",
        help="Directory where highlight clips and the reel will be saved.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="YAML",
        help="Path to a YAML configuration file (optional).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        metavar="SCORE",
        help="Minimum highlight score [0–1] to generate a clip (default: 0.7).",
    )
    parser.add_argument(
        "--before",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds of padding before each highlight (default: 5).",
    )
    parser.add_argument(
        "--after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds of padding after each highlight (default: 5).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio analysis (faster, but audio score will be 0).",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip pose estimation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="Optional file to write log output to.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Main processing pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    setup_logging(args.log_level, args.log_file)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config: AppConfig = load_config(args.config)

    # CLI overrides
    if args.threshold is not None:
        config.highlight.score_threshold = args.threshold
    if args.before is not None:
        config.highlight.clip_before = args.before
    if args.after is not None:
        config.highlight.clip_after = args.after

    logger.info("=== Pyckleball Highlight Extractor ===")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Score threshold: %.2f", config.highlight.score_threshold)

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    loader = VideoLoader(args.input)
    meta = loader.metadata

    # ------------------------------------------------------------------
    # Audio analysis (run before frame loop – works on entire file)
    # ------------------------------------------------------------------
    audio = AudioAnalyzer(config.audio)
    if not args.no_audio:
        logger.info("Detecting audio excitement…")
        audio.analyze(args.input)

    # ------------------------------------------------------------------
    # Initialise vision components
    # ------------------------------------------------------------------
    ball_tracker = BallTracker(config.detection)
    player_tracker = PlayerTracker(config.detection)
    pose_estimator = PoseEstimator(config.detection.device) if not args.no_pose else None
    rally_detector = RallyDetector(config.rally)

    # ------------------------------------------------------------------
    # Main frame processing loop
    # ------------------------------------------------------------------
    logger.info("Detecting rallies… (%d frames @ %.1f fps)", meta.total_frames, meta.fps)
    start_time = time.time()

    rallies = []
    with tqdm(
        total=meta.total_frames,
        unit="frame",
        desc="Processing",
        dynamic_ncols=True,
    ) as pbar:
        for frame_idx, timestamp, frame in loader.frames():
            # Ball detection
            ball_det = ball_tracker.process_frame(frame_idx, timestamp, frame)

            # Player tracking
            player_dets = player_tracker.process_frame(frame_idx, timestamp, frame)
            frame_count_so_far = frame_idx + 1
            movement_score = player_tracker.get_combined_movement_score(
                frame_count_so_far
            )

            # Pose estimation (every 5th frame to reduce compute)
            if pose_estimator is not None and frame_idx % 5 == 0:
                pose_result = pose_estimator.process_frame(frame_idx, timestamp, frame)
                # Celebration / raised paddle slightly boosts movement score
                if pose_result and pose_result.event in (
                    PoseEvent.CELEBRATION,
                    PoseEvent.RAISED_PADDLE,
                ):
                    movement_score = min(movement_score + 0.1, 1.0)

            # Rally detection
            ball_state = ball_tracker.get_state()
            completed_rally = rally_detector.update(
                timestamp, ball_det, ball_state, movement_score
            )
            if completed_rally is not None:
                rallies.append(completed_rally)

            pbar.update(1)

    # Flush any active rally
    final_rally = rally_detector.flush(meta.duration)
    if final_rally is not None:
        rallies.append(final_rally)

    elapsed = time.time() - start_time
    logger.info(
        "Frame processing complete: %d rallies detected in %.1fs",
        len(rallies),
        elapsed,
    )

    if not rallies:
        logger.warning("No rallies detected. Check that the video contains gameplay.")
        return 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    logger.info("Scoring %d rally candidates…", len(rallies))
    scorer = HighlightScorer(config.scoring, audio)
    candidates = scorer.score_rallies(rallies)

    above_threshold = [c for c in candidates if c.score >= config.highlight.score_threshold]
    logger.info(
        "%d/%d candidates above threshold %.2f",
        len(above_threshold),
        len(candidates),
        config.highlight.score_threshold,
    )

    # ------------------------------------------------------------------
    # Clip generation
    # ------------------------------------------------------------------
    clips_dir = os.path.join(args.output, "clips")
    generator = ClipGenerator(config.highlight, clips_dir)
    clips = generator.generate(
        candidates,
        args.input,
        meta,
        threshold=config.highlight.score_threshold,
    )

    # ------------------------------------------------------------------
    # Highlight compilation
    # ------------------------------------------------------------------
    compiler = HighlightCompiler(config.highlight, args.output)

    if clips:
        reel_path = compiler.compile(clips)
        if reel_path:
            logger.info("Highlight reel: %s", reel_path)
    else:
        logger.warning("No clips were generated.")

    meta_path = compiler.write_metadata(clips)
    logger.info("Metadata: %s", meta_path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if pose_estimator is not None:
        pose_estimator.close()

    logger.info("Done. Generated %d highlight clip(s).", len(clips))
    return 0


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
