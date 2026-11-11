"""Đo chất lượng bộ gán nhãn so với nhãn gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knee_mri.config import Config
from knee_mri.constants import LABELS
from knee_mri.data.catalog import StudyCatalog
from knee_mri.labeling import cached_weak_labels, format_labeler_report, score_labeler


def test_score_labeler_counts_confusion(catalog: StudyCatalog) -> None:
    uid = catalog.studies_with_gold()[0]
    truth = catalog.gold(uid)
    assert truth is not None
    # Đoán đúng mọi nhãn → không có FP/FN.
    report = score_labeler(catalog, lambda _u: dict(truth), [uid])
    assert report.studies == 1
    assert report.micro.fp == 0 and report.micro.fn == 0
    assert report.micro.tp == sum(int(truth.get(n, 0)) for n in LABELS)
    # Đoán ngược toàn bộ → không có TP/TN.
    flipped = score_labeler(
        catalog, lambda _u: {n: 1 - int(truth.get(n, 0)) for n in LABELS}, [uid]
    )
    assert flipped.micro.tp == 0 and flipped.micro.tn == 0


def test_score_labeler_skips_studies_without_gold(catalog: StudyCatalog) -> None:
    no_gold = next(u for u in catalog.studies() if catalog.gold(u) is None)
    assert score_labeler(catalog, lambda _u: {}, [no_gold]).studies == 0


def test_cached_weak_labels_reads_json_and_defaults_to_empty(cfg: Config, tmp_path: Path) -> None:
    cfg.paths.weak_labels.mkdir(parents=True, exist_ok=True)
    (cfg.paths.weak_labels / "s1.json").write_text(json.dumps({"ACL": 1}))
    predict = cached_weak_labels(cfg)
    assert predict("s1")["ACL"] == 1
    assert predict("missing") == {}


def test_format_report_lists_every_label(catalog: StudyCatalog) -> None:
    text = format_labeler_report(score_labeler(catalog, lambda _u: {}))
    for name in LABELS:
        assert name in text
    assert "macro-F1" in text


@pytest.mark.parametrize("precision_inputs", [(0, 0, 0, 5)])
def test_f1_is_zero_when_nothing_predicted(precision_inputs: tuple[int, ...]) -> None:
    from knee_mri.labeling.evaluate import LabelScore

    assert LabelScore("x", *precision_inputs).f1 == 0.0
