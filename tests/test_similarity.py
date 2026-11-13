"""Kiểm thử các phép đo độ giống nhau giữa vector cache."""

from __future__ import annotations

import numpy as np
import pytest

from knee_mri.config import load_config
from knee_mri.training.similarity import run_similarity, similarity_report


def _near_identical(rng: np.random.Generator, n: int = 40, d: int = 32) -> np.ndarray:
    base = rng.normal(size=d) * 10
    return base + rng.normal(size=(n, d)) * 0.01


class TestSimilarityReport:
    def test_identical_vectors_are_flagged(self) -> None:
        rng = np.random.default_rng(0)
        report = similarity_report("x", _near_identical(rng))
        assert report.mean_cosine > 0.99
        assert report.kd_floor < 0.01
        assert "ĐÚNG" in report.verdict() and "KHÔNG" not in report.verdict()

    def test_independent_vectors_are_not_flagged(self) -> None:
        rng = np.random.default_rng(0)
        report = similarity_report("x", rng.normal(size=(200, 64)))
        assert abs(report.mean_cosine) < 0.1
        assert report.kd_floor > 0.9
        assert report.effective_rank > 30
        assert "KHÔNG" in report.verdict()

    def test_common_direction_is_separated_from_signal(self) -> None:
        """Lệch hướng chung làm cosine thô cao nhưng cosine sau khi trừ mean thấp."""
        rng = np.random.default_rng(1)
        report = similarity_report("x", 20.0 + rng.normal(size=(200, 64)))
        assert report.mean_cosine > 0.95
        assert report.mean_cosine_centered < 0.1
        assert report.effective_rank > 30  # phần dư trải đều, không sụp về một hướng

    def test_matches_naive_pairwise_computation(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(size=(12, 8)) + 1.0
        unit = data / np.linalg.norm(data, axis=1, keepdims=True)
        gram = unit @ unit.T
        naive = (gram.sum() - np.trace(gram)) / (12 * 11)
        assert similarity_report("x", data).mean_cosine == pytest.approx(naive)

    def test_rejects_too_few_vectors(self) -> None:
        with pytest.raises(ValueError):
            similarity_report("x", np.zeros((2, 4)))
        with pytest.raises(ValueError):
            similarity_report("x", np.zeros(4))


class TestRunSimilarity:
    def test_missing_cache_is_skipped(self, tmp_path) -> None:
        cfg = load_config("test", overrides={"paths": {"artifacts_root": str(tmp_path)}})
        assert run_similarity(cfg) == []

    def test_reads_guidance_and_teacher_dirs(self, tmp_path) -> None:
        cfg = load_config("test", overrides={"paths": {"artifacts_root": str(tmp_path)}})
        rng = np.random.default_rng(3)
        cfg.paths.guidance.mkdir(parents=True)
        for index in range(5):
            np.save(cfg.paths.guidance / f"s{index}.npy", rng.normal(size=16).astype(np.float32))
        study = cfg.paths.teacher_feats / "study0"
        study.mkdir(parents=True)
        for index in range(4):
            np.save(study / f"series{index}.npy", rng.normal(size=8).astype(np.float32))

        reports = run_similarity(cfg)
        assert [r.name for r in reports] == ["guidance (VLM)", "teacher (3DINO)"]
        assert [r.count for r in reports] == [5, 4]
        assert [r.dim for r in reports] == [16, 8]
