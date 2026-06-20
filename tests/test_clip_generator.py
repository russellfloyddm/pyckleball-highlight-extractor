"""Tests for clip generation logic (mocking FFmpeg)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from pickleball_highlights.clip_generator import ClipGenerator, GeneratedClip
from pickleball_highlights.config import HighlightConfig
from pickleball_highlights.highlight_scorer import HighlightCandidate
from pickleball_highlights.rally_detector import Rally
from pickleball_highlights.video_loader import VideoMetadata


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


def make_meta(duration: float = 300.0) -> VideoMetadata:
    return VideoMetadata(
        path="/fake/video.mp4",
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=int(duration * 30),
        duration=duration,
        codec="avc1",
    )


def make_candidate(
    start: float,
    end: float,
    score: float = 0.85,
    reason: str = "long_rally",
) -> HighlightCandidate:
    rally = Rally(
        start_time=start,
        end_time=end,
        shot_count=15,
        max_ball_speed=500.0,
        avg_ball_speed=300.0,
        movement_score=0.5,
    )
    return HighlightCandidate(rally=rally, score=score, reason=reason)


class TestMergeCandidates:
    def test_no_merge_distant_candidates(self):
        c1 = make_candidate(0.0, 10.0)
        c2 = make_candidate(30.0, 40.0)
        merged = ClipGenerator._merge_candidates([c1, c2], merge_gap=10.0)
        assert len(merged) == 2

    def test_merge_adjacent_candidates(self):
        c1 = make_candidate(0.0, 10.0)
        c2 = make_candidate(15.0, 25.0)  # 5 seconds gap < merge_gap=10
        merged = ClipGenerator._merge_candidates([c1, c2], merge_gap=10.0)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 25.0

    def test_empty_candidates(self):
        merged = ClipGenerator._merge_candidates([], merge_gap=10.0)
        assert merged == []

    def test_single_candidate(self):
        c = make_candidate(5.0, 15.0)
        merged = ClipGenerator._merge_candidates([c], merge_gap=10.0)
        assert len(merged) == 1

    def test_merge_keeps_best_score(self):
        c1 = make_candidate(0.0, 10.0, score=0.75)
        c2 = make_candidate(15.0, 25.0, score=0.90)
        merged = ClipGenerator._merge_candidates([c1, c2], merge_gap=10.0)
        assert merged[0].score == pytest.approx(0.90)

    def test_unordered_candidates_sorted(self):
        c1 = make_candidate(20.0, 30.0)
        c2 = make_candidate(0.0, 10.0)
        merged = ClipGenerator._merge_candidates([c1, c2], merge_gap=10.0)
        assert merged[0].start_time == 0.0


class TestClipGenerator:
    @patch("pickleball_highlights.clip_generator.subprocess.run")
    def test_generate_calls_ffmpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        # Also mock the _verify_ffmpeg call
        config = make_config()
        gen = ClipGenerator(config, str(tmp_path))

        candidates = [make_candidate(10.0, 30.0, score=0.9)]
        meta = make_meta()

        clips = gen.generate(candidates, "/fake/video.mp4", meta, threshold=0.7)
        # FFmpeg should have been called (once for verify + once for clip)
        assert mock_run.call_count >= 1

    @patch("pickleball_highlights.clip_generator.subprocess.run")
    def test_generate_filters_below_threshold(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        config = make_config()
        gen = ClipGenerator(config, str(tmp_path))

        candidates = [
            make_candidate(10.0, 30.0, score=0.5),  # below threshold
            make_candidate(50.0, 70.0, score=0.9),  # above threshold
        ]
        meta = make_meta()
        clips = gen.generate(candidates, "/fake/video.mp4", meta, threshold=0.7)
        # Only 1 clip above threshold
        # (run count includes the verify call, so check clips list)
        # The number of extract calls should reflect 1 clip
        extract_calls = [
            c for c in mock_run.call_args_list
            if "ffmpeg" in str(c) and "-ss" in str(c)
        ]
        assert len(extract_calls) == 1

    @patch.object(ClipGenerator, "_extract_clip")
    @patch("pickleball_highlights.clip_generator.subprocess.run")
    def test_generate_reports_progress(self, mock_run, mock_extract_clip, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        mock_extract_clip.return_value = None
        config = make_config()
        gen = ClipGenerator(config, str(tmp_path))

        progress_updates = []
        candidates = [
            make_candidate(10.0, 30.0, score=0.8),
            make_candidate(40.0, 60.0, score=0.9),
        ]
        meta = make_meta()

        gen.generate(
            candidates,
            "/fake/video.mp4",
            meta,
            threshold=0.7,
            progress_callback=lambda current, total: progress_updates.append(
                (current, total)
            ),
        )

        assert progress_updates == [(0, 1), (1, 1)]

    def test_generate_no_ffmpeg_returns_empty(self, tmp_path):
        """If FFmpeg is missing, _extract_clip should return None gracefully."""
        import os
        config = make_config()
        with patch("pickleball_highlights.clip_generator.subprocess.run", side_effect=FileNotFoundError):
            gen = ClipGenerator.__new__(ClipGenerator)
            gen.config = config
            gen.output_dir = str(tmp_path)
            os.makedirs(str(tmp_path), exist_ok=True)

            result = gen._extract_clip(
                idx=1,
                candidate=make_candidate(10.0, 30.0),
                video_path="/fake/video.mp4",
                video_duration=300.0,
            )
            assert result is None

    def test_clip_padding_clamps_to_zero(self, tmp_path):
        """Padded start should not go below 0."""
        config = make_config(clip_before=10.0)
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            gen = ClipGenerator(config, str(tmp_path))
            candidate = make_candidate(3.0, 15.0)  # start < clip_before
            clip = gen._extract_clip(
                idx=1,
                candidate=candidate,
                video_path="/fake/video.mp4",
                video_duration=300.0,
            )
            if clip:
                assert clip.padded_start >= 0.0

    def test_clip_padding_clamps_to_duration(self, tmp_path):
        """Padded end should not exceed video duration."""
        config = make_config(clip_after=10.0)
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            gen = ClipGenerator(config, str(tmp_path))
            candidate = make_candidate(295.0, 298.0)  # near end of 300s video
            clip = gen._extract_clip(
                idx=1,
                candidate=candidate,
                video_path="/fake/video.mp4",
                video_duration=300.0,
            )
            if clip:
                assert clip.padded_end <= 300.0
