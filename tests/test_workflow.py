"""Test các hàm hỗ trợ cho notebook: pilot, preflight, runner, đánh giá gold, CAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config, load_config  # noqa: E402
from knee_mri.data.catalog import SeriesInfo, StudyCatalog  # noqa: E402
from knee_mri.data.dataset import missing_cache  # noqa: E402
from knee_mri.data.series_selection import select_series  # noqa: E402
from knee_mri.data.splits import make_split, pilot_subset  # noqa: E402
from knee_mri.inference.predictor import Predictor, evaluate_gold  # noqa: E402
from knee_mri.pipeline import (  # noqa: E402
    StageTimer,
    compute_cam,
    estimate_full_run,
    estimate_mask_gb,
    run_preflight,
)
from knee_mri.training.checkpoint import BEST_NAME, CheckpointState, save_checkpoint  # noqa: E402
from knee_mri.training.trainer import build_bundle  # noqa: E402
from knee_mri.utils.io import save_json, save_npy  # noqa: E402


@pytest.fixture
def big_catalog() -> StudyCatalog:
    """20 study, gold nằm rải rác ở cuối — mô phỏng việc cắt [:N] làm mất gold."""
    uids = [f"s{i:02d}" for i in range(20)]
    series = {u: [SeriesInfo(f"{u}-1", "Sagittal", 1, 1)] for u in uids}
    reports = dict.fromkeys(uids, "")
    gold = {u: {"ACL": i % 2} for i, u in enumerate(uids) if i >= 15}
    return StudyCatalog(series, reports, gold, uids)


class TestPilotSubset:
    def test_keeps_every_gold_study(self, big_catalog: StudyCatalog) -> None:
        subset = pilot_subset(big_catalog, extra=3, seed=1)
        assert set(big_catalog.studies_with_gold()) <= set(subset)
        assert len(subset) == 5 + 3

    def test_plain_truncation_would_lose_gold(self, big_catalog: StudyCatalog) -> None:
        """Đối chứng: đây chính là lỗi của ``LIMIT = 50``."""
        assert make_split(big_catalog, limit=10).val == []
        assert len(make_split(big_catalog, subset=pilot_subset(big_catalog, extra=3)).val) == 5

    def test_is_reproducible_and_clamps_extra(self, big_catalog: StudyCatalog) -> None:
        assert pilot_subset(big_catalog, extra=4, seed=9) == pilot_subset(
            big_catalog, extra=4, seed=9
        )
        assert len(pilot_subset(big_catalog, extra=999)) == 20
        assert len(pilot_subset(big_catalog, extra=-5)) == 5

    def test_split_subset_is_disjoint_and_restricted(self, big_catalog: StudyCatalog) -> None:
        subset = pilot_subset(big_catalog, extra=3, seed=1)
        split = make_split(big_catalog, subset=subset)
        assert set(split.train) | set(split.val) == set(subset)
        assert not set(split.train) & set(split.val)


class TestMissingCache:
    def test_reports_gaps_then_clears(self, cfg: Config) -> None:
        cfg.paths.ensure()
        uids = ["a", "b"]
        assert missing_cache(cfg, uids) == {"teacher_feats": 2, "guidance": 2, "weak_labels": 2}

        for uid in uids:
            save_npy(
                np.zeros(cfg.model.teacher_dim, np.float32), cfg.paths.teacher_feats / uid / "s.npy"
            )
            save_npy(
                np.zeros(cfg.model.guidance_dim, np.float32), cfg.paths.guidance / f"{uid}.npy"
            )
        save_json({}, cfg.paths.weak_labels / "a.json")
        assert missing_cache(cfg, uids) == {"weak_labels": 1}
        save_json({}, cfg.paths.weak_labels / "b.json")
        assert missing_cache(cfg, uids) == {}


class TestStageTimer:
    def test_returns_result_and_records_time(self) -> None:
        timer = StageTimer()
        assert timer.run("x", lambda a, b=1: a + b, 1, b=2) == 3
        assert timer.seconds["x"] >= 0

    def test_retries_transient_errors(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("OOM tạm thời")
            return "ok"

        assert StageTimer().run("x", flaky, retries=2) == "ok"
        assert calls["n"] == 3

    def test_gives_up_after_retries(self) -> None:
        calls = {"n": 0}

        def always() -> None:
            calls["n"] += 1
            raise RuntimeError("hỏng")

        with pytest.raises(RuntimeError):
            StageTimer().run("x", always, retries=1)
        assert calls["n"] == 2

    def test_does_not_retry_config_errors(self) -> None:
        calls = {"n": 0}

        def broken() -> None:
            calls["n"] += 1
            raise ImportError("thiếu module")

        with pytest.raises(ImportError):
            StageTimer().run("x", broken, retries=3)
        assert calls["n"] == 1


class TestEstimateFullRun:
    def test_scales_linearly_by_study_count(self) -> None:
        hours = estimate_full_run(
            {"masks": 3600.0}, pilot_studies=100, full_studies=1000, full_epochs=30
        )
        assert hours["masks"] == pytest.approx(10.0)
        assert hours["total"] == pytest.approx(10.0)

    def test_includes_training(self) -> None:
        hours = estimate_full_run(
            {"masks": 0.0},
            pilot_studies=10,
            full_studies=10,
            epoch_seconds=60.0,
            pilot_train=10,
            full_train=100,
            full_epochs=30,
        )
        assert hours["train"] == pytest.approx(60 / 10 * 100 * 30 / 3600)

    def test_rejects_empty_pilot(self) -> None:
        with pytest.raises(ValueError):
            estimate_full_run({}, pilot_studies=0, full_studies=1)


class TestPreflight:
    def test_estimate_mask_gb_uses_target_shape(self, cfg: Config, catalog: StudyCatalog) -> None:
        d, h, w = cfg.data.target_shape
        uids = ["study-a", "study-b"]
        n_series = sum(len(select_series(catalog, uid, cfg.data)) for uid in uids)
        assert n_series >= 2
        expected = n_series * d * h * w / 1e9
        assert estimate_mask_gb(cfg, catalog, ["study-a", "study-b"]) == pytest.approx(expected)

    def test_flags_missing_dino_assets(self, cfg: Config, catalog: StudyCatalog) -> None:
        report = run_preflight(cfg, catalog, ["study-a"], require_cuda=False, check_hf=False)
        assert not report.ok
        assert any("3DINO" in problem for problem in report.problems)
        with pytest.raises(RuntimeError, match="Preflight"):
            report.raise_if_failed()

    def test_passes_when_assets_exist(self, cfg: Config, catalog: StudyCatalog) -> None:
        cfg.paths.ensure()
        cfg.paths.dino_code_dir.mkdir(parents=True, exist_ok=True)
        cfg.dino.weights_path(cfg.paths.artifacts_root).write_bytes(b"x")
        report = run_preflight(cfg, catalog, ["study-a"], require_cuda=False, check_hf=False)
        assert report.ok, report.problems
        assert report.info["free_gb"] > 0


class TestEvaluateAndCam:
    @pytest.fixture
    def predictor(self, cfg: Config) -> Predictor:
        cfg.paths.ensure()
        bundle = build_bundle(cfg)
        path = cfg.paths.checkpoints / BEST_NAME
        save_checkpoint(CheckpointState(epoch=1, best_metric=0.5, modules=bundle.as_dict()), path)
        return Predictor.from_checkpoint(cfg, checkpoint=path, device="cpu")

    def test_evaluate_gold_none_without_gold(
        self, predictor: Predictor, catalog: StudyCatalog
    ) -> None:
        assert evaluate_gold(predictor, catalog, ["study-b"]) is None

    def test_compute_cam_none_without_series(
        self, cfg: Config, predictor: Predictor, catalog
    ) -> None:
        # Không có DICOM thật cho catalog giả → series_dir None → trả None, không crash.
        assert compute_cam(cfg, predictor, catalog, "study-c") is None
        assert compute_cam(cfg, predictor, catalog, "study-a") is None


class TestDinoWeightsOverride:
    """Notebook 02 dùng đường dẫn tuyệt đối thay symlink (Google Drive không có symlink)."""

    def test_absolute_weight_file_ignores_artifacts_root(self, tmp_path: Path) -> None:
        shared = tmp_path / "artifacts" / "3dino_vit_weights.pth"
        cfg = load_config(
            "test",
            overrides={
                "paths": {"artifacts_root": str(tmp_path / "artifacts" / "pilot")},
                "dino": {"weight_file": str(shared)},
            },
        )
        assert cfg.dino.weights_path(cfg.paths.artifacts_root) == shared
