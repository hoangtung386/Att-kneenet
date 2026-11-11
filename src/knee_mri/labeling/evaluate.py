"""Đo chất lượng một bộ gán nhãn yếu so với nhãn gold.

Weak label là tín hiệu huấn luyện cho 4349 study không có nhãn thật, nên chất
lượng của nó giới hạn trần của model. Hàm ở đây cho phép đo **cùng một cách** cho
luật từ khóa lẫn VLM (qua cache đã precompute), để chọn nguồn nhãn có căn cứ
thay vì tin vào cảm giác.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from knee_mri.config.schema import Config
from knee_mri.constants import LABELS
from knee_mri.data.catalog import StudyCatalog
from knee_mri.utils.io import load_json

LabelFn = Callable[[str], Mapping[str, int | float]]


@dataclass(frozen=True, slots=True)
class LabelScore:
    """Ma trận nhầm lẫn của một nhãn."""

    label: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass(frozen=True, slots=True)
class LabelerReport:
    """Điểm theo từng nhãn và tổng gộp (micro) của một bộ gán nhãn."""

    per_label: list[LabelScore]
    studies: int

    @property
    def micro(self) -> LabelScore:
        return LabelScore(
            "micro",
            sum(s.tp for s in self.per_label),
            sum(s.fp for s in self.per_label),
            sum(s.fn for s in self.per_label),
            sum(s.tn for s in self.per_label),
        )

    @property
    def macro_f1(self) -> float:
        return sum(s.f1 for s in self.per_label) / max(len(self.per_label), 1)


def score_labeler(
    catalog: StudyCatalog,
    predict: LabelFn,
    study_uids: Sequence[str] | None = None,
    *,
    labels: Sequence[str] = LABELS,
) -> LabelerReport:
    """Chấm ``predict(study_uid) -> {nhãn: 0/1}`` trên các study có nhãn gold.

    Args:
        predict: Hàm nhận ``study_uid`` và trả nhãn dự đoán (giá trị > 0,5 = dương).
        study_uids: Tập study cần chấm; mặc định toàn bộ study có gold.
    """
    uids = list(study_uids) if study_uids is not None else catalog.studies_with_gold()
    counts = {name: [0, 0, 0, 0] for name in labels}  # tp, fp, fn, tn
    scored = 0
    for uid in uids:
        truth = catalog.gold(uid)
        if truth is None:
            continue
        scored += 1
        predicted = predict(uid)
        for name in labels:
            t, p = bool(truth.get(name, 0)), float(predicted.get(name, 0)) > 0.5
            counts[name][(0 if t else 1) if p else (2 if t else 3)] += 1
    return LabelerReport([LabelScore(n, *counts[n]) for n in labels], scored)


def cached_weak_labels(cfg: Config) -> LabelFn:
    """Bộ đọc nhãn yếu đã precompute (luật hoặc VLM) trong ``artifacts/weak_labels``.

    Study chưa có file được coi là toàn 0, nên chỉ chấm sau khi precompute xong
    (``missing_cache`` báo thiếu), nếu không điểm sẽ thấp giả tạo.
    """

    def predict(study_uid: str) -> Mapping[str, int | float]:
        path = cfg.paths.weak_labels / f"{study_uid}.json"
        return load_json(path) if path.is_file() else {}

    return predict


def format_labeler_report(report: LabelerReport) -> str:
    """Bảng precision/recall/F1 theo nhãn, dễ đọc trong notebook."""
    rows = [f"{'nhãn':<18}{'dương':>6}{'prec':>7}{'rec':>7}{'F1':>7}"]
    for s in [*report.per_label, report.micro]:
        rows.append(f"{s.label:<18}{s.positives:>6}{s.precision:>7.2f}{s.recall:>7.2f}{s.f1:>7.2f}")
    rows.append(f"macro-F1 = {report.macro_f1:.3f} trên {report.studies} study gold")
    return "\n".join(rows)


__all__ = [
    "LabelScore",
    "LabelerReport",
    "cached_weak_labels",
    "format_labeler_report",
    "score_labeler",
]
