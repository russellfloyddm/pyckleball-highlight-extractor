"""Audio analysis for detecting crowd excitement and volume spikes."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pickleball_highlights.config import AudioConfig
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AudioFrame:
    """Audio analysis result for one time window."""

    start_time: float  # seconds
    end_time: float  # seconds
    rms_energy: float  # root-mean-square energy
    excitement_score: float  # normalised [0, 1]
    is_spike: bool


class AudioAnalyzer:
    """Analyse audio track to detect excitement spikes.

    Uses ``librosa`` for spectral analysis.  Falls back to a simple RMS
    energy approach via ``numpy`` when ``librosa`` is not available.

    Args:
        config: Audio analysis configuration.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._frames: List[AudioFrame] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> List[AudioFrame]:
        """Extract and analyse the audio track from a video file.

        Extracts audio to a temporary WAV file using FFmpeg, then runs
        spectral analysis.

        Args:
            video_path: Path to the source video file.

        Returns:
            List of AudioFrame objects covering the full video duration.
        """
        logger.info("Analysing audio track from %s", os.path.basename(video_path))
        self._frames.clear()

        wav_path = self._extract_audio(video_path)
        if wav_path is None:
            logger.warning("Audio extraction failed – skipping audio analysis.")
            return []

        try:
            frames = self._process_wav(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        self._frames = frames
        logger.info(
            "Audio analysis complete: %d frames, %d spikes detected",
            len(frames),
            sum(1 for f in frames if f.is_spike),
        )
        return frames

    def get_score_at(self, timestamp: float) -> float:
        """Return the audio excitement score at a given timestamp.

        Args:
            timestamp: Time in seconds.

        Returns:
            Excitement score in [0, 1].  Returns 0 if timestamp is outside
            the analysed range or audio analysis was not performed.
        """
        for frame in self._frames:
            if frame.start_time <= timestamp < frame.end_time:
                return frame.excitement_score
        return 0.0

    def get_frames(self) -> List[AudioFrame]:
        """Return all audio analysis frames."""
        return list(self._frames)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Use FFmpeg to extract audio as a WAV file.

        Returns:
            Path to the temporary WAV file, or None on failure.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", str(self.config.sample_rate),
            "-vn",
            wav_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning(
                    "FFmpeg audio extraction failed (code %d): %s",
                    result.returncode,
                    result.stderr.decode(errors="replace")[:500],
                )
                os.unlink(wav_path)
                return None
            return wav_path
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found – audio analysis disabled.  Install FFmpeg "
                "and ensure it is on PATH."
            )
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio extraction timed out.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Audio extraction error: %s", exc)

        try:
            os.unlink(wav_path)
        except OSError:
            pass
        return None

    def _process_wav(self, wav_path: str) -> List[AudioFrame]:
        """Analyse a WAV file and return per-window AudioFrame objects."""
        try:
            import librosa  # type: ignore

            return self._librosa_analysis(wav_path)
        except ImportError:
            logger.warning("librosa not installed – using basic RMS analysis.")
            return self._numpy_rms_analysis(wav_path)

    def _librosa_analysis(self, wav_path: str) -> List[AudioFrame]:
        """Full spectral analysis using librosa."""
        import librosa  # type: ignore

        y, sr = librosa.load(wav_path, sr=self.config.sample_rate, mono=True)
        window_samples = int(self.config.frame_duration * sr)
        hop_samples = window_samples

        frames: List[AudioFrame] = []
        rms_values: List[float] = []

        for i in range(0, len(y) - window_samples + 1, hop_samples):
            chunk = y[i : i + window_samples]
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_values.append(rms)

        if not rms_values:
            return []

        rms_arr = np.array(rms_values, dtype=np.float32)
        mean_rms = float(np.mean(rms_arr))
        std_rms = float(np.std(rms_arr))
        # Normalise to [0, 1] using z-score clamping
        if std_rms > 0:
            z_scores = (rms_arr - mean_rms) / std_rms
        else:
            z_scores = np.zeros_like(rms_arr)

        excite = np.clip((z_scores + 2) / 4, 0, 1)  # map z in [-2,+2] → [0,1]

        spike_threshold = mean_rms + 1.5 * std_rms

        for i, rms in enumerate(rms_values):
            start = i * self.config.frame_duration
            end = start + self.config.frame_duration
            frames.append(
                AudioFrame(
                    start_time=start,
                    end_time=end,
                    rms_energy=rms,
                    excitement_score=float(excite[i]),
                    is_spike=rms > spike_threshold,
                )
            )
        return frames

    def _numpy_rms_analysis(self, wav_path: str) -> List[AudioFrame]:
        """Lightweight RMS-only analysis using numpy + wave module."""
        import wave

        try:
            with wave.open(wav_path, "r") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
        except Exception as exc:
            logger.warning("Could not read WAV file: %s", exc)
            return []

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        window_samples = int(self.config.frame_duration * framerate)
        hop_samples = window_samples
        frames: List[AudioFrame] = []
        rms_values: List[float] = []

        for i in range(0, len(samples) - window_samples + 1, hop_samples):
            chunk = samples[i : i + window_samples]
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_values.append(rms)

        if not rms_values:
            return []

        rms_arr = np.array(rms_values, dtype=np.float32)
        mean_rms = float(np.mean(rms_arr))
        std_rms = float(np.std(rms_arr))
        spike_threshold = mean_rms + 1.5 * std_rms

        if std_rms > 0:
            z_scores = (rms_arr - mean_rms) / std_rms
        else:
            z_scores = np.zeros_like(rms_arr)
        excite = np.clip((z_scores + 2) / 4, 0, 1)

        for i, rms in enumerate(rms_values):
            start = i * self.config.frame_duration
            end = start + self.config.frame_duration
            frames.append(
                AudioFrame(
                    start_time=start,
                    end_time=end,
                    rms_energy=float(rms),
                    excitement_score=float(excite[i]),
                    is_spike=float(rms) > spike_threshold,
                )
            )
        return frames
