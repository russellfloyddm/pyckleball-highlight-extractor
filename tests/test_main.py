"""Tests for the main CLI entry point."""

from __future__ import annotations

import argparse
import pytest
from unittest.mock import MagicMock, patch

from pickleball_highlights.main import parse_args


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
