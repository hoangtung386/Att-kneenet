"""Test baseline probe trên đặc trưng teacher đã cache."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config  # noqa: E402
from knee_mri.constants import LABELS  # noqa: E402
from knee_mri.data.catalog import SeriesInfo, StudyCatalog  # noqa: E402
from knee_mri.training.probe import (  # noqa: E402
    fit_probe,
    load_probe_data,
    run_probe,
    score_probe,
    study_teacher_feature,
)
from knee_mri.utils.io import save_json, save_npy  # noqa: E402


def _synthetic(n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = np.stack([(features[:, i] > 0).astype(np.float32) for i in range(len(LABELS))], 1)
    return features, labels


class TestFitProbe:
    def test_learns_a_separable_signal(self) -> None:
        train_x, train_y = _synthetic(300, 32, 0)
        val_x, val_y = _synthetic(100, 32, 1)
        model, stats = fit_probe(train_x, train_y)
        assert score_probe(model, stats, val_x, val_y).macro > 0.9

    def test_is_near_chance_on_pure_noise(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 32)).astype(np.float32)
        y = (rng.random((300, 12)) > 0.5).astype(np.float32)
        model, stats = fit_probe(x[:200], y[:200], epochs=50)
        auc = score_probe(model, stats, x[200:], y[200:]).macro
        assert 0.35 < auc < 0.65


@pytest.fixture
def cached(cfg: Config) -> tuple[StudyCatalog, list[str]]:
    """Catalog nhỏ với đặc trưng và nhãn yếu ghi sẵn vào artifacts của test."""
    cfg.paths.ensure()
    uids = [f"study-{i}" for i in range(30)]
    series = {uid: [SeriesInfo(f"{uid}-s", "Sagittal", 1, 1)] for uid in uids}
    gold = {uid: dict.fromkeys(LABELS, i % 2) for i, uid in enumerate(uids[:6])}
    catalog = StudyCatalog(series, dict.fromkeys(uids, "r"), gold, uids)
    features, labels = _synthetic(len(uids), cfg.model.teacher_dim, 0)
    for uid, vector, row in zip(uids, features, labels, strict=True):
        save_npy(vector, cfg.paths.teacher_feats / uid / f"{uid}-s.npy")
        save_json(
            dict(zip(LABELS, row.tolist(), strict=True)), cfg.paths.weak_labels / f"{uid}.json"
        )
    return catalog, uids


class TestLoadProbeData:
    def test_weak_target_reads_cached_labels(self, cfg: Config, cached) -> None:
        catalog, uids = cached
        data = load_probe_data(cfg, catalog, uids, target="weak")
        assert data.features.shape == (30, cfg.model.teacher_dim)
        assert data.labels.shape == (30, 12)

    def test_gold_target_keeps_only_gold_studies(self, cfg: Config, cached) -> None:
        catalog, uids = cached
        assert len(load_probe_data(cfg, catalog, uids, target="gold").study_uids) == 6

    def test_skips_studies_without_features(self, cfg: Config, cached) -> None:
        catalog, _ = cached
        assert study_teacher_feature(cfg, catalog, "khong-co") is None

    def test_rejects_unknown_target(self, cfg: Config, cached) -> None:
        catalog, uids = cached
        with pytest.raises(ValueError):
            load_probe_data(cfg, catalog, uids, target="khac")

    def test_run_probe_reports_none_when_data_is_missing(self, cfg: Config) -> None:
        empty = StudyCatalog({}, {}, {}, [])
        assert run_probe(cfg, empty, []) is None
