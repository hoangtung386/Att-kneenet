"""Baseline probe trên đặc trưng 3DINO đã cache — phép thử "dữ liệu có đủ tốt không".

Trước khi tốn hàng chục giờ GPU huấn luyện student, cần biết liệu đặc trưng
3DINO (đã tính sẵn ở bước ``teacher-feats``) cộng nhãn yếu có dự đoán được nhãn
gold hay không. Probe này huấn luyện một hồi quy logistic đa nhãn trên vector
trung bình các series của study, mất vài giây trên CPU:

* AUC trên gold **cao** (≳ 0.65): nhãn và đặc trưng tốt, việc cần làm là cải
  thiện student/huấn luyện.
* AUC trên gold **≈ 0.5**: vấn đề nằm ở nhãn yếu hoặc đặc trưng, huấn luyện thêm
  student sẽ không cứu được.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from knee_mri.config.schema import Config
from knee_mri.constants import LABELS, NUM_LABELS
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.series_selection import select_series
from knee_mri.training.losses import bce_loss
from knee_mri.training.metrics import AucReport, macro_auc
from knee_mri.utils.io import load_json, load_npy
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeData:
    """Đặc trưng và nhãn mức study, đã loại các study thiếu cache."""

    study_uids: list[str]
    features: np.ndarray  # (n, teacher_dim)
    labels: np.ndarray  # (n, NUM_LABELS)


def study_teacher_feature(cfg: Config, catalog: StudyCatalog, study_uid: str) -> np.ndarray | None:
    """Trung bình đặc trưng teacher qua các series đã chọn, ``None`` nếu thiếu cache."""
    vectors = []
    for series in select_series(catalog, study_uid, cfg.data):
        path = cfg.paths.teacher_feats / study_uid / f"{series.series_uid}.npy"
        if path.is_file():
            vectors.append(load_npy(path))
    return np.mean(vectors, axis=0) if vectors else None


def load_probe_data(
    cfg: Config, catalog: StudyCatalog, study_uids: Sequence[str], *, target: str
) -> ProbeData:
    """Gom đặc trưng teacher và nhãn cho các study cho trước.

    Args:
        target: ``"weak"`` đọc nhãn yếu đã precompute, ``"gold"`` đọc nhãn người gán.
    """
    if target not in {"weak", "gold"}:
        raise ValueError(f"target phải là 'weak' hoặc 'gold', nhận được {target!r}.")

    kept: list[str] = []
    features: list[np.ndarray] = []
    labels: list[list[float]] = []
    for uid in study_uids:
        feature = study_teacher_feature(cfg, catalog, uid)
        if feature is None:
            continue
        if target == "gold":
            truth = catalog.gold(uid)
            if truth is None:
                continue
            row = [float(truth.get(name, 0)) for name in LABELS]
        else:
            path = cfg.paths.weak_labels / f"{uid}.json"
            if not path.is_file():
                continue
            scores = load_json(path)
            row = [float(scores.get(name, 0)) for name in LABELS]
        kept.append(uid)
        features.append(feature)
        labels.append(row)

    dim = cfg.model.teacher_dim
    return ProbeData(
        kept,
        np.stack(features) if features else np.zeros((0, dim), dtype=np.float32),
        np.asarray(labels, dtype=np.float32).reshape(-1, NUM_LABELS),
    )


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> tuple[nn.Linear, tuple[Tensor, Tensor]]:
    """Huấn luyện hồi quy logistic đa nhãn full-batch.

    Đặc trưng được chuẩn hóa z-score theo thống kê của tập train; trả kèm
    ``(mean, std)`` để áp đúng phép chuẩn hóa đó lên tập đánh giá.
    """
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std

    model = nn.Linear(x.shape[1], y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        bce_loss(model(x), y).backward()
        optimizer.step()
    return model, (mean, std)


@torch.no_grad()
def score_probe(
    model: nn.Linear, stats: tuple[Tensor, Tensor], features: np.ndarray, labels: np.ndarray
) -> AucReport:
    """AUC macro của probe trên một tập có nhãn."""
    mean, std = stats
    x = (torch.from_numpy(np.asarray(features, dtype=np.float32)) - mean) / std
    return macro_auc(labels, torch.sigmoid(model(x)).numpy())


def run_probe(cfg: Config, catalog: StudyCatalog, study_uids: Sequence[str]) -> AucReport | None:
    """Huấn luyện trên nhãn yếu của study không gold, đo AUC trên study gold.

    Returns:
        :class:`AucReport`, hoặc ``None`` khi thiếu cache/nhãn để chạy.
    """
    gold_uids = set(catalog.studies_with_gold())
    train = load_probe_data(
        cfg, catalog, [u for u in study_uids if u not in gold_uids], target="weak"
    )
    val = load_probe_data(cfg, catalog, [u for u in study_uids if u in gold_uids], target="gold")
    if len(train.study_uids) < 10 or not val.study_uids:
        logger.error(
            "Không đủ dữ liệu cho probe: %d study train, %d study gold có đặc trưng teacher.",
            len(train.study_uids),
            len(val.study_uids),
        )
        return None
    logger.info(
        "Probe: %d study train (nhãn yếu), %d study gold.",
        len(train.study_uids),
        len(val.study_uids),
    )
    model, stats = fit_probe(train.features, train.labels, seed=cfg.seed)
    return score_probe(model, stats, val.features, val.labels)


__all__ = [
    "ProbeData",
    "fit_probe",
    "load_probe_data",
    "run_probe",
    "score_probe",
    "study_teacher_feature",
]
