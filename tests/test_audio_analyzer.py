"""Tests for the audio analyzer module."""

from __future__ import annotations

import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from pickleball_highlights.audio_analyzer import AudioAnalyzer, AudioFrame
from pickleball_highlights.config import AudioConfig


def write_test_wav(path: str, duration: float = 5.0, sample_rate: int = 22050) -> None:
    """Write a simple sine-wave WAV for testing."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


class TestAudioAnalyzerNumpy:
    """Tests using the numpy RMS fallback (no librosa required)."""

    def _make_analyzer(self) -> AudioAnalyzer:
        config = AudioConfig(sample_rate=22050, frame_duration=1.0)
        return AudioAnalyzer(config)

    def test_process_wav_numpy(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        write_test_wav(wav_path, duration=5.0)

        analyzer = self._make_analyzer()
        # Call the private method directly to bypass FFmpeg dependency
        frames = analyzer._numpy_rms_analysis(wav_path)

        assert len(frames) == 5
        for frame in frames:
            assert isinstance(frame, AudioFrame)
            assert frame.rms_energy >= 0.0
            assert 0.0 <= frame.excitement_score <= 1.0
            assert isinstance(frame.is_spike, bool)

    def test_get_score_at_returns_zero_before_analysis(self):
        analyzer = self._make_analyzer()
        score = analyzer.get_score_at(5.0)
        assert score == 0.0

    def test_get_frames_empty_before_analysis(self):
        analyzer = self._make_analyzer()
        assert analyzer.get_frames() == []

    def test_get_score_at_after_manual_injection(self):
        analyzer = self._make_analyzer()
        # Manually inject frames for testing
        analyzer._frames = [
            AudioFrame(start_time=0.0, end_time=1.0, rms_energy=0.5, excitement_score=0.6, is_spike=False),
            AudioFrame(start_time=1.0, end_time=2.0, rms_energy=0.8, excitement_score=0.9, is_spike=True),
        ]
        assert analyzer.get_score_at(0.5) == pytest.approx(0.6)
        assert analyzer.get_score_at(1.5) == pytest.approx(0.9)

    def test_get_score_at_out_of_range_returns_zero(self):
        analyzer = self._make_analyzer()
        analyzer._frames = [
            AudioFrame(start_time=0.0, end_time=1.0, rms_energy=0.5, excitement_score=0.6, is_spike=False),
        ]
        assert analyzer.get_score_at(99.0) == 0.0

    def test_spike_detection(self, tmp_path):
        """Frames with high RMS should be flagged as spikes."""
        wav_path = str(tmp_path / "spike.wav")
        sample_rate = 22050
        duration = 4.0
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        # Quiet signal with a loud burst in the last second
        signal = np.zeros_like(t)
        signal[: int(3 * sample_rate)] = np.sin(2 * np.pi * 440 * t[: int(3 * sample_rate)]) * 100
        signal[int(3 * sample_rate) :] = np.sin(2 * np.pi * 440 * t[int(3 * sample_rate) :]) * 30000

        pcm = signal.astype(np.int16)
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        config = AudioConfig(sample_rate=sample_rate, frame_duration=1.0)
        analyzer = AudioAnalyzer(config)
        frames = analyzer._numpy_rms_analysis(wav_path)

        assert len(frames) == 4
        # Last frame should be a spike
        assert frames[-1].is_spike

    def test_empty_wav_returns_no_frames(self, tmp_path):
        wav_path = str(tmp_path / "empty.wav")
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"")

        analyzer = self._make_analyzer()
        frames = analyzer._numpy_rms_analysis(wav_path)
        assert frames == []
