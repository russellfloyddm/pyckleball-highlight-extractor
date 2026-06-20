"""Tests for the configuration module."""

import pytest
import yaml
import os
import tempfile

from pickleball_highlights.config import (
    AppConfig,
    AudioConfig,
    DetectionConfig,
    HighlightConfig,
    RallyConfig,
    ScoringWeights,
    load_config,
)


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        total = w.rally_duration + w.shot_count + w.ball_speed + w.movement + w.audio
        assert abs(total - 1.0) < 1e-6

    def test_custom_valid_weights(self):
        w = ScoringWeights(
            rally_duration=0.4,
            shot_count=0.3,
            ball_speed=0.1,
            movement=0.1,
            audio=0.1,
        )
        assert abs(w.rally_duration + w.shot_count + w.ball_speed + w.movement + w.audio - 1.0) < 1e-6

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(
                rally_duration=0.5,
                shot_count=0.5,
                ball_speed=0.5,
                movement=0.5,
                audio=0.5,
            )


class TestAppConfig:
    def test_default_config(self):
        cfg = AppConfig()
        assert isinstance(cfg.scoring, ScoringWeights)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.rally, RallyConfig)
        assert isinstance(cfg.highlight, HighlightConfig)
        assert cfg.log_level == "INFO"

    def test_default_threshold(self):
        cfg = AppConfig()
        assert cfg.highlight.score_threshold == 0.7

    def test_default_clip_padding(self):
        cfg = AppConfig()
        assert cfg.highlight.clip_before == 5.0
        assert cfg.highlight.clip_after == 5.0

    def test_default_merge_gap(self):
        cfg = AppConfig()
        assert cfg.highlight.merge_gap == 10.0

    def test_default_rally_filters(self):
        cfg = AppConfig()
        assert cfg.rally.min_rally_duration == 3.0
        assert cfg.rally.min_shots == 3
        assert cfg.rally.min_detection_streak == 3
        assert cfg.rally.detection_window == 5
        assert cfg.rally.min_movement_to_start == 0.05
        assert cfg.rally.min_ball_speed == 20.0


class TestLoadConfig:
    def test_load_nonexistent_returns_defaults(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, AppConfig)

    def test_load_none_returns_defaults(self):
        cfg = load_config(None)
        assert isinstance(cfg, AppConfig)

    def test_load_yaml_overrides(self, tmp_path):
        data = {
            "log_level": "DEBUG",
            "scoring": {
                "rally_duration": 0.40,
                "shot_count": 0.30,
                "ball_speed": 0.10,
                "movement": 0.10,
                "audio": 0.10,
            },
            "highlight": {
                "score_threshold": 0.8,
                "clip_before": 3.0,
                "clip_after": 3.0,
                "merge_gap": 10.0,
                "output_format": "mp4",
                "ffmpeg_crf": 23,
                "ffmpeg_preset": "medium",
                "add_transitions": True,
                "transition_duration": 0.5,
                "title_screen": False,
                "title_text": "Test",
                "title_duration": 3.0,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(data))

        cfg = load_config(str(config_file))
        assert cfg.log_level == "DEBUG"
        assert cfg.scoring.rally_duration == 0.40
        assert cfg.highlight.score_threshold == 0.8
        assert cfg.highlight.clip_before == 3.0

    def test_load_yaml_long_rally_bonus_int_keys(self, tmp_path):
        data = {
            "rally": {
                "min_rally_duration": 2.0,
                "max_gap_duration": 3.0,
                "min_shots": 2,
                "min_detection_streak": 2,
                "detection_window": 4,
                "min_movement_to_start": 0.01,
                "min_ball_speed": 10.0,
                "service_anchor_window": 2.0,
                "service_anchor_position_std": 6.0,
                "service_anchor_min_speed_spike": 60.0,
                "long_rally_bonus_shots": {10: 0.1, 20: 0.2, 30: 0.35},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(data))

        cfg = load_config(str(config_file))
        assert 10 in cfg.rally.long_rally_bonus_shots
        assert cfg.rally.long_rally_bonus_shots[30] == 0.35
        assert cfg.rally.min_detection_streak == 2
        assert cfg.rally.detection_window == 4
        assert cfg.rally.min_movement_to_start == 0.01
        assert cfg.rally.min_ball_speed == 10.0
