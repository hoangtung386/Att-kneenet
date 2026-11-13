"""Test vòng huấn luyện: scheduler, validation, checkpoint và resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config, load_config  # noqa: E402
from knee_mri.constants import LABELS  # noqa: E402
from knee_mri.data.catalog import StudyCatalog  # noqa: E402
from knee_mri.data.dataset import KneeDataset  # noqa: E402
from knee_mri.training.checkpoint import BEST_NAME, LAST_NAME  # noqa: E402
from knee_mri.training.trainer import (  # noqa: E402
    EpochStats,
    build_bundle,
    build_optimizer,
    build_scheduler,
    train,
)


class TestScheduler:
    def _lrs(self, cfg: Config, steps_per_epoch: int) -> list[float]:
        bundle = build_bundle(cfg)
        optimizer = build_optimizer(bundle, cfg)
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
        lrs = []
        for _ in range(steps_per_epoch * cfg.train.epochs):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        return lrs

    def test_warms_up_then_decays_to_zero(self, cfg: Config) -> None:
        lrs = self._lrs(replace(cfg, train=replace(cfg.train, epochs=10)), 10)
        peak = max(lrs)
        assert lrs[0] == 0.0
        assert peak == pytest.approx(cfg.train.lr, rel=0.05)
        assert lrs.index(peak) < len(lrs) // 4
        assert lrs[-1] < 0.01 * peak


class TestOptimizerGroups:
    def test_no_weight_decay_on_bias_norm_and_cls(self, cfg: Config) -> None:
        optimizer = build_optimizer(build_bundle(cfg), cfg)
        decay, no_decay = optimizer.param_groups
        assert decay["weight_decay"] == cfg.train.weight_decay
        assert no_decay["weight_decay"] == 0.0
        assert all(p.ndim >= 2 for p in decay["params"])


class TestEpochStats:
    def test_averages_over_steps(self) -> None:
        stats = EpochStats()
        stats.update(total=2.0, kd=1.0, contrast=0.0, cls=1.0)
        stats.update(total=4.0, kd=3.0, contrast=0.0, cls=1.0)
        assert stats.averaged()["total"] == pytest.approx(3.0)
        assert stats.averaged()["kd"] == pytest.approx(2.0)

    def test_empty_epoch_does_not_divide_by_zero(self) -> None:
        assert EpochStats().averaged()["total"] == 0.0


@pytest.fixture(scope="module")
def real_catalog() -> StudyCatalog:
    cfg = load_config("test")
    if not cfg.paths.train_csv.is_file():
        pytest.skip("Không có dataset cục bộ.")
    return StudyCatalog.from_csv(cfg.paths.train_csv, cfg.paths.train_series_csv)


@pytest.fixture
def tiny_run(cfg: Config, real_catalog: StudyCatalog) -> tuple[KneeDataset, KneeDataset]:
    """4 study train + 4 study val; nhãn gold gán giả cho val để AUC đo được."""
    uids = real_catalog.studies_with_series()[:8]
    gold = {
        uid: {name: (i + j) % 2 for j, name in enumerate(LABELS)} for i, uid in enumerate(uids[4:])
    }
    catalog = StudyCatalog(
        {uid: real_catalog.series(uid) for uid in uids},
        {uid: real_catalog.report(uid) for uid in uids},
        gold,
        uids,
    )
    kwargs = {"use_dummy_teacher": True, "use_dummy_guidance": True}
    return (
        KneeDataset(uids[:4], catalog, cfg, augment=True, **kwargs),
        KneeDataset(uids[4:], catalog, cfg, **kwargs),
    )


class TestTrainLoop:
    def test_runs_validates_and_writes_checkpoints(
        self, cfg: Config, tiny_run: tuple[KneeDataset, KneeDataset]
    ) -> None:
        train_ds, val_ds = tiny_run
        result = train(cfg, train_ds, val_ds, device="cpu")
        assert result.history and "val_macro_auc" in result.history[0]
        assert 0.0 <= result.best_metric <= 1.0
        assert (cfg.paths.checkpoints / BEST_NAME).is_file()
        assert (cfg.paths.checkpoints / LAST_NAME).is_file()

    def test_without_validation_still_leaves_a_best_checkpoint(
        self, cfg: Config, tiny_run: tuple[KneeDataset, KneeDataset]
    ) -> None:
        result = train(cfg, tiny_run[0], None, device="cpu")
        assert result.best_checkpoint.is_file()

    def test_resume_continues_after_the_saved_epoch(
        self, cfg: Config, tiny_run: tuple[KneeDataset, KneeDataset]
    ) -> None:
        train(cfg, *tiny_run, device="cpu")
        longer = replace(cfg, train=replace(cfg.train, epochs=cfg.train.epochs + 1))
        result = train(longer, *tiny_run, device="cpu", resume=True)
        assert len(result.history) == 1  # chỉ chạy epoch còn thiếu
        assert result.history[0]["epoch"] == float(cfg.train.epochs)

    def test_checkpoint_loads_for_inference(
        self, cfg: Config, tiny_run: tuple[KneeDataset, KneeDataset], tmp_path: Path
    ) -> None:
        from knee_mri.inference.predictor import Predictor

        train(cfg, *tiny_run, device="cpu")
        predictor = Predictor.from_checkpoint(cfg, device="cpu")
        probabilities = predictor.predict_study(tiny_run[1].catalog, tiny_run[1].study_uids[0])
        assert probabilities.shape == (len(LABELS),)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
