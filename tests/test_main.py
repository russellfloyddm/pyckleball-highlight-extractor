"""Tests for the main CLI entry point."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from pickleball_highlights.config import AppConfig
from pickleball_highlights.main import parse_args, run
from pickleball_highlights.rally_detector import Rally


class TestParseArgs:
    def test_required_input_and_output(self):
        args = parse_args(["--input", "match.mp4", "--output", "./highlights"])
        assert args.input == "match.mp4"
        assert args.output == "./highlights"

    def test_short_flags(self):
        args = parse_args(["-i", "match.mp4", "-o", "./out"])
        assert args.input == "match.mp4"
        assert args.output == "./out"

    def test_threshold_override(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--threshold", "0.8"])
        assert args.threshold == pytest.approx(0.8)

    def test_before_after_override(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--before", "3", "--after", "7"])
        assert args.before == pytest.approx(3.0)
        assert args.after == pytest.approx(7.0)

    def test_start_end_time_override(self):
        args = parse_args(
            ["-i", "v.mp4", "-o", "./out", "--start-time", "30", "--end-time", "45.5"]
        )
        assert args.start_time == pytest.approx(30.0)
        assert args.end_time == pytest.approx(45.5)

    def test_no_audio_flag(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--no-audio"])
        assert args.no_audio is True

    def test_no_pose_flag(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--no-pose"])
        assert args.no_pose is True

    def test_log_level(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_config_path(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out", "--config", "my.yaml"])
        assert args.config == "my.yaml"

    def test_missing_input_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--output", "./out"])

    def test_missing_output_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--input", "v.mp4"])

    def test_defaults(self):
        args = parse_args(["-i", "v.mp4", "-o", "./out"])
        assert args.threshold is None
        assert args.before is None
        assert args.after is None
        assert args.no_audio is False
        assert args.no_pose is False
        assert args.config is None
        assert args.log_level == "INFO"
        assert args.start_time is None
        assert args.end_time is None


class RecordingObserver:
    def __init__(self):
        self.status = []
        self.progress = []

    def on_status(self, stage: str, message: str) -> None:
        self.status.append((stage, message))

    def on_progress(self, stage: str, current: int, total: int) -> None:
        self.progress.append((stage, current, total))


class FakeTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount):
        pass


class TestRunObserver:
    @patch("pickleball_highlights.main.HighlightCompiler")
    @patch("pickleball_highlights.main.ClipGenerator")
    @patch("pickleball_highlights.main.HighlightScorer")
    @patch("pickleball_highlights.main.RallyDetector")
    @patch("pickleball_highlights.main.PlayerTracker")
    @patch("pickleball_highlights.main.BallTracker")
    @patch("pickleball_highlights.main.AudioAnalyzer")
    @patch("pickleball_highlights.main.VideoLoader")
    @patch("pickleball_highlights.main.load_config")
    @patch("pickleball_highlights.main.setup_logging")
    @patch("pickleball_highlights.main.tqdm", FakeTqdm)
    def test_run_reports_progress(
        self,
        mock_setup_logging,
        mock_load_config,
        mock_video_loader_cls,
        mock_audio_cls,
        mock_ball_tracker_cls,
        mock_player_tracker_cls,
        mock_rally_detector_cls,
        mock_scorer_cls,
        mock_generator_cls,
        mock_compiler_cls,
        tmp_path,
    ):
        args = argparse.Namespace(
            input="match.mp4",
            output=str(tmp_path),
            config=None,
            threshold=None,
            before=None,
            after=None,
            start_time=12.0,
            end_time=20.0,
            no_audio=True,
            no_pose=True,
            log_level="INFO",
            log_file=None,
        )
        observer = RecordingObserver()

        mock_load_config.return_value = AppConfig()
        mock_video_loader = MagicMock()
        mock_video_loader.metadata = SimpleNamespace(
            total_frames=3,
            fps=30.0,
            duration=0.1,
        )
        mock_video_loader.frames.return_value = iter(
            [
                (0, 0.0, MagicMock()),
                (1, 0.033, MagicMock()),
                (2, 0.066, MagicMock()),
            ]
        )
        mock_video_loader_cls.return_value = mock_video_loader

        mock_audio_cls.return_value = MagicMock()
        mock_ball_tracker = MagicMock()
        mock_ball_tracker.get_state.return_value = MagicMock()
        mock_ball_tracker_cls.return_value = mock_ball_tracker

        mock_player_tracker = MagicMock()
        mock_player_tracker.get_combined_movement_score.return_value = 0.0
        mock_player_tracker_cls.return_value = mock_player_tracker

        rally = Rally(
            start_time=0.0,
            end_time=5.0,
            shot_count=12,
            max_ball_speed=450.0,
            avg_ball_speed=250.0,
            movement_score=0.4,
        )
        mock_rally_detector = MagicMock()
        mock_rally_detector.update.return_value = None
        mock_rally_detector.flush.return_value = rally
        mock_rally_detector_cls.return_value = mock_rally_detector

        candidate = SimpleNamespace(score=0.9)
        mock_scorer = MagicMock()
        mock_scorer.score_rallies.return_value = [candidate]
        mock_scorer_cls.return_value = mock_scorer

        mock_generator = MagicMock()

        def generate_with_progress(*args, **kwargs):
            kwargs["progress_callback"](1, 1)
            return ["clip-1"]

        mock_generator.generate.side_effect = generate_with_progress
        mock_generator_cls.return_value = mock_generator

        mock_compiler = MagicMock()
        mock_compiler.compile.return_value = "reel.mp4"
        mock_compiler.write_metadata.return_value = "metadata.json"
        mock_compiler_cls.return_value = mock_compiler

        exit_code = run(args, observer=observer)

        assert exit_code == 0
        mock_video_loader_cls.assert_called_once_with(
            "match.mp4", start_time=12.0, end_time=20.0
        )
        assert ("audio", 1, 1) in observer.progress
        assert ("frames", 3, 3) in observer.progress
        assert ("clips", 1, 1) in observer.progress
        assert observer.status[-1] == ("done", "Generated 1 highlight clip(s)")


class TestRunTimeWindowValidation:
    def test_negative_start_time_raises(self, tmp_path):
        args = argparse.Namespace(
            input="match.mp4",
            output=str(tmp_path),
            config=None,
            threshold=None,
            before=None,
            after=None,
            start_time=-1.0,
            end_time=None,
            no_audio=True,
            no_pose=True,
            log_level="INFO",
            log_file=None,
        )
        with pytest.raises(ValueError, match="--start-time must be >= 0"):
            run(args)

    def test_end_time_before_start_time_raises(self, tmp_path):
        args = argparse.Namespace(
            input="match.mp4",
            output=str(tmp_path),
            config=None,
            threshold=None,
            before=None,
            after=None,
            start_time=30.0,
            end_time=10.0,
            no_audio=True,
            no_pose=True,
            log_level="INFO",
            log_file=None,
        )
        with pytest.raises(
            ValueError, match="--end-time must be greater than --start-time"
        ):
            run(args)
