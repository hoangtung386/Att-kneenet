"""Đo mức giống nhau giữa các vector đã cache — kiểm chứng hai giả thuyết trong
``docs/root_cause_analysis.md``:

* §4.2 — vector **guidance** của các study gần như giống hệt nhau (cosine trung
  bình > 0.95), nên InfoNCE không có gì để phân biệt.
* §4.5 — đặc trưng **teacher** 3DINO của các gối MRI có cosine rất cao với nhau,
  nên KD thắng "miễn phí" bằng cách xuất ra vector trung bình.

Ba số cho mỗi tập vector:

``mean_cosine``
    cosine trung bình giữa mọi cặp khác nhau. Gần 1 = các vector gần như trùng.
``mean_cosine_centered``
    như trên sau khi trừ vector trung bình toàn tập, tức chỉ còn phần riêng của
    từng mẫu. Số này **không** tự cho biết phần riêng đó là tín hiệu hay nhiễu;
    đọc cùng ``effective_rank`` và thử thực nghiệm (probe, InfoNCE trên vector
    đã trừ mean) trước khi kết luận.
``mean_predictor_cosine``
    cosine trung bình giữa mỗi vector và vector trung bình. ``1 - số này`` là
    KD loss mà một student **chỉ xuất ra vector trung bình** đạt được; nếu loss
    KD thực tế không thấp hơn đáng kể thì student chưa học gì từ ảnh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knee_mri.config.schema import Config
from knee_mri.utils.io import load_npy
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

# Ngưỡng dùng để phát biểu kết luận; xem docstring module cho ý nghĩa.
HIGH_SIMILARITY: float = 0.95
MIN_VECTORS: int = 3


@dataclass(frozen=True, slots=True)
class SimilarityReport:
    """Thống kê độ giống nhau của một tập vector."""

    name: str
    count: int
    dim: int
    mean_cosine: float
    mean_cosine_centered: float
    mean_predictor_cosine: float
    effective_rank: float
    top_component_share: float

    @property
    def kd_floor(self) -> float:
        """KD loss đạt được khi chỉ xuất vector trung bình (``1 - cosine``)."""
        return 1.0 - self.mean_predictor_cosine

    def verdict(self) -> str:
        """Kết luận một câu theo tiêu chí của §4.2/§4.5: cosine thô ≥ ``HIGH_SIMILARITY``."""
        if self.mean_cosine >= HIGH_SIMILARITY:
            return (
                "các vector gần như trùng nhau: GIẢ THUYẾT ĐÚNG. Thông tin riêng từng mẫu "
                "chỉ nằm trong phần sai khác nhỏ sau khi trừ vector trung bình"
            )
        return "đủ khác nhau giữa các mẫu: giả thuyết KHÔNG đúng"

    def format(self) -> str:
        return (
            f"{self.name}: n={self.count} dim={self.dim}\n"
            f"  cosine trung bình (thô)         = {self.mean_cosine:.4f}\n"
            f"  cosine trung bình (đã trừ mean) = {self.mean_cosine_centered:.4f}\n"
            f"  cosine tới vector trung bình    = {self.mean_predictor_cosine:.4f}"
            f"  (KD loss sàn = {self.kd_floor:.4f})\n"
            f"  hạng hiệu dụng                  = {self.effective_rank:.1f}"
            f"  | thành phần chính 1 chiếm {self.top_component_share:.1%}\n"
            f"  => {self.verdict()}"
        )


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _mean_off_diagonal(unit: np.ndarray) -> float:
    """Cosine trung bình giữa mọi cặp (i, j), i ≠ j, từ các hàng đã chuẩn hóa."""
    count = unit.shape[0]
    total = unit.sum(axis=0)
    # sum_{i != j} <u_i, u_j> = |sum u|^2 - sum |u_i|^2, không cần dựng ma trận n x n.
    cross = float(total @ total - (unit * unit).sum())
    return cross / (count * (count - 1))


def similarity_report(name: str, vectors: np.ndarray) -> SimilarityReport:
    """Tính :class:`SimilarityReport` cho ma trận ``(n, d)``.

    Raises:
        ValueError: nếu ``vectors`` không phải ma trận 2 chiều hoặc ít hơn
            ``MIN_VECTORS`` hàng (số liệu không có nghĩa).
    """
    if vectors.ndim != 2 or vectors.shape[0] < MIN_VECTORS:
        raise ValueError(f"{name}: cần ma trận (n >= {MIN_VECTORS}, d), nhận {vectors.shape}")
    data = vectors.astype(np.float64)
    mean = data.mean(axis=0)

    unit = _normalise(data)
    raw = _mean_off_diagonal(unit)
    centered = _mean_off_diagonal(_normalise(data - mean))
    predictor = float((unit @ (mean / max(np.linalg.norm(mean), 1e-12))).mean())

    singular = np.linalg.svd(data - mean, compute_uv=False)
    energy = singular**2
    share = energy / max(energy.sum(), 1e-12)
    positive = share[share > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum())) if positive.size else 0.0

    return SimilarityReport(
        name=name,
        count=data.shape[0],
        dim=data.shape[1],
        mean_cosine=raw,
        mean_cosine_centered=centered,
        mean_predictor_cosine=predictor,
        effective_rank=effective_rank,
        top_component_share=float(share[0]) if share.size else 0.0,
    )


def _load_dir(root: Path, *, limit: int | None) -> np.ndarray | None:
    files = sorted(root.rglob("*.npy"))
    if limit is not None:
        files = files[:limit]
    if not files:
        return None
    return np.stack([np.asarray(load_npy(path), dtype=np.float32).ravel() for path in files])


def run_similarity(cfg: Config, *, limit: int | None = None) -> list[SimilarityReport]:
    """Đo guidance và đặc trưng teacher trong ``artifacts``; bỏ qua tập nào thiếu cache."""
    reports: list[SimilarityReport] = []
    sources: Sequence[tuple[str, Path]] = (
        ("guidance (VLM)", cfg.paths.guidance),
        ("teacher (3DINO)", cfg.paths.teacher_feats),
    )
    for name, root in sources:
        vectors = _load_dir(root, limit=limit) if root.is_dir() else None
        if vectors is None or vectors.shape[0] < MIN_VECTORS:
            logger.warning("%s: không có đủ file .npy trong %s — bỏ qua.", name, root)
            continue
        reports.append(similarity_report(name, vectors))
    return reports


__all__ = ["HIGH_SIMILARITY", "SimilarityReport", "run_similarity", "similarity_report"]
