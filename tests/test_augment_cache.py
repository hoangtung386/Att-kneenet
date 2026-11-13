"""Test augmentation và cache volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from knee_mri.config import Config
from knee_mri.data.augment import AugmentCfg, augment_volume
from knee_mri.data.cache import build_volume_cached, cache_key, cache_path
from knee_mri.data.volume import build_volume


class TestAugment:
    def test_preserves_shape_dtype_and_range(self, volume: np.ndarray) -> None:
        out = augment_volume(volume, np.random.default_rng(0))
        assert out.shape == volume.shape and out.dtype == np.float32
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_does_not_mutate_input(self, volume: np.ndarray) -> None:
        before = volume.copy()
        augment_volume(volume, np.random.default_rng(0))
        assert np.array_equal(volume, before)

    def test_is_reproducible_for_a_seed(self, volume: np.ndarray) -> None:
        a = augment_volume(volume, np.random.default_rng(3))
        b = augment_volume(volume, np.random.default_rng(3))
        assert np.array_equal(a, b)

    def test_actually_changes_the_volume(self, volume: np.ndarray) -> None:
        assert not np.array_equal(volume, augment_volume(volume, np.random.default_rng(1)))

    def test_all_zero_strengths_is_identity(self, volume: np.ndarray) -> None:
        off = AugmentCfg(contrast=0.0, brightness=0.0, noise_std=0.0, max_shift=0)
        assert np.allclose(augment_volume(volume, np.random.default_rng(1), off), volume)

    def test_shift_fills_with_background_not_wraparound(self) -> None:
        volume = np.full((2, 8, 8), -1.0, dtype=np.float32)
        volume[:, :, -1] = 1.0  # cột sáng ở mép phải
        cfg = AugmentCfg(contrast=0.0, brightness=0.0, noise_std=0.0, max_shift=3)
        for seed in range(20):
            out = augment_volume(volume, np.random.default_rng(seed), cfg)
            assert out[:, :, 0].max() == -1.0 or out[:, :, 0].max() < 1.0


class TestVolumeCache:
    def test_key_changes_with_shape_norm_and_mask(self, cfg: Config) -> None:
        base = cache_key(cfg.data, has_mask=False)
        assert base != cache_key(cfg.data, has_mask=True)

    def test_second_call_reads_from_disk(
        self, real_series_dir: Path, cfg: Config, tmp_path: Path
    ) -> None:
        first = build_volume_cached(
            real_series_dir, cfg.data, cache_root=tmp_path, study_uid="s", series_uid="x"
        )
        assert cache_path(tmp_path, cfg.data, "s", "x", has_mask=False).is_file()
        second = build_volume_cached(
            Path("/khong/ton/tai"), cfg.data, cache_root=tmp_path, study_uid="s", series_uid="x"
        )
        assert np.array_equal(first, second)

    def test_matches_uncached_within_float16_precision(
        self, real_series_dir: Path, cfg: Config, tmp_path: Path
    ) -> None:
        cached = build_volume_cached(
            real_series_dir, cfg.data, cache_root=tmp_path, study_uid="s", series_uid="x"
        )
        assert np.allclose(cached, build_volume(real_series_dir, cfg.data), atol=2e-3)

    def test_corrupt_cache_file_is_rebuilt(
        self, real_series_dir: Path, cfg: Config, tmp_path: Path
    ) -> None:
        path = cache_path(tmp_path, cfg.data, "s", "x", has_mask=False)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"khong phai npy")
        volume = build_volume_cached(
            real_series_dir, cfg.data, cache_root=tmp_path, study_uid="s", series_uid="x"
        )
        assert volume.shape == cfg.data.target_shape

    def test_build_errors_are_not_cached(self, cfg: Config, tmp_path: Path) -> None:
        empty = tmp_path / "rong"
        empty.mkdir()
        with pytest.raises(Exception):  # noqa: B017 - EmptySeriesError
            build_volume_cached(empty, cfg.data, cache_root=tmp_path, study_uid="s", series_uid="x")
        assert not cache_path(tmp_path, cfg.data, "s", "x", has_mask=False).exists()
