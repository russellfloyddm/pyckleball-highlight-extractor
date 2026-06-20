"""Tests for the highlight compiler module."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from pickleball_highlights.clip_generator import GeneratedClip
from pickleball_highlights.config import HighlightConfig
from pickleball_highlights.highlight_compiler import HighlightCompiler


def make_config(**kwargs) -> HighlightConfig:
    defaults = dict(
        score_threshold=0.7,
        merge_gap=10.0,
        clip_before=5.0,
        clip_after=5.0,
        output_format="mp4",
        ffmpeg_crf=23,
        ffmpeg_preset="medium",
        add_transitions=True,
        transition_duration=0.5,
        title_screen=False,
        title_text="Test",
        title_duration=3.0,
    )
    defaults.update(kwargs)
    return HighlightConfig(**defaults)


def make_clip(idx: int, tmp_path, score: float = 0.85) -> GeneratedClip:
    clip_path = str(tmp_path / f"highlight_{idx:03d}.mp4")
    # Create a dummy file so FFmpeg concat list is valid
    open(clip_path, "w").close()
    return GeneratedClip(
        index=idx,
        source_path="/fake/video.mp4",
        output_path=clip_path,
        start_time=float(idx * 30),
        end_time=float(idx * 30 + 20),
        padded_start=float(idx * 30 - 5),
        padded_end=float(idx * 30 + 25),
        duration=30.0,
        score=score,
        reason="long_rally",
    )


class TestHighlightCompiler:
    def test_write_metadata_creates_json(self, tmp_path):
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clips = [make_clip(1, tmp_path), make_clip(2, tmp_path)]

        meta_path = compiler.write_metadata(clips, filename="metadata.json")

        assert os.path.isfile(meta_path)
        with open(meta_path, "r") as fh:
            data = json.load(fh)

        assert data["total_clips"] == 2
        assert len(data["highlights"]) == 2

    def test_metadata_fields(self, tmp_path):
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clip = make_clip(1, tmp_path, score=0.92)
        meta_path = compiler.write_metadata([clip])

        with open(meta_path, "r") as fh:
            data = json.load(fh)

        h = data["highlights"][0]
        assert h["index"] == 1
        assert h["score"] == pytest.approx(0.92, abs=1e-3)
        assert h["reason"] == "long_rally"
        assert "start" in h
        assert "end" in h
        assert "duration" in h

    def test_compile_no_clips_returns_none(self, tmp_path):
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        result = compiler.compile([])
        assert result is None

    @patch("pickleball_highlights.highlight_compiler.subprocess.run")
    def test_compile_calls_ffmpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clips = [make_clip(1, tmp_path), make_clip(2, tmp_path)]

        result = compiler.compile(clips, output_filename="reel.mp4")
        assert mock_run.called

    @patch("pickleball_highlights.highlight_compiler.subprocess.run")
    def test_compile_ffmpeg_failure_returns_none(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr=b"error")
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clips = [make_clip(1, tmp_path)]

        result = compiler.compile(clips)
        assert result is None

    @patch("pickleball_highlights.highlight_compiler.subprocess.run", side_effect=FileNotFoundError)
    def test_compile_no_ffmpeg_returns_none(self, mock_run, tmp_path):
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clips = [make_clip(1, tmp_path)]

        result = compiler.compile(clips)
        assert result is None

    def test_write_concat_list_creates_file(self, tmp_path):
        config = make_config()
        compiler = HighlightCompiler(config, str(tmp_path))
        clips = [make_clip(1, tmp_path), make_clip(2, tmp_path)]

        list_path = compiler._write_concat_list(clips)
        assert list_path is not None
        assert os.path.isfile(list_path)

        with open(list_path) as fh:
            content = fh.read()
        assert "file" in content
        assert "highlight_001.mp4" in content

        os.unlink(list_path)
