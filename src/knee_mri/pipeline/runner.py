"""Chạy từng bước pipeline có đo thời gian, tự thử lại và ngoại suy chi phí.

Notebook Colab chạy các bước dài (SAM, VLM, 3DINO) trên Drive, nơi lỗi thoáng qua
như OOM tạm thời hay mất kết nối là chuyện thường. Mọi bước precompute đều bỏ
qua file đã có, nên thử lại chỉ làm phần còn thiếu — không mất công đã làm.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import torch

from knee_mri.models.teachers.dino3d import DinoUnavailableError
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

#: Lỗi do cấu hình/môi trường — thử lại chỉ tốn thời gian, nên dừng ngay.
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    ImportError,
    AssertionError,
    DinoUnavailableError,
    KeyboardInterrupt,
)

#: Các khóa không phải bước theo-từng-study khi ngoại suy.
TRAIN_KEY = "train"


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class StageTimer:
    """Chạy các bước theo tên, cộng dồn thời gian (kể cả các lần thử lại)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def run(
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        retries: int = 2,
        non_retryable: tuple[type[BaseException], ...] = NON_RETRYABLE,
        **kwargs: Any,
    ) -> T:
        """Gọi ``fn(*args, **kwargs)``; lỗi thoáng qua được thử lại tối đa ``retries`` lần."""
        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except non_retryable:
                self._add(name, start)
                raise
            except Exception as exc:  # noqa: BLE001 - mọi lỗi runtime đều có thể là thoáng qua
                self._add(name, start)
                logger.warning(
                    "[%s] lỗi lần %d/%d: %s: %s",
                    name,
                    attempt + 1,
                    retries + 1,
                    type(exc).__name__,
                    exc,
                )
                _free_gpu_memory()
                if attempt == retries:
                    raise
                logger.warning("[%s] thử lại — file đã làm xong sẽ được bỏ qua.", name)
                continue
            self._add(name, start)
            logger.info("[%s] xong sau %.1f phút", name, self.seconds[name] / 60)
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    def _add(self, name: str, start: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def estimate_full_run(
    stage_seconds: Mapping[str, float],
    *,
    pilot_studies: int,
    full_studies: int,
    epoch_seconds: float | None = None,
    pilot_train: int | None = None,
    full_train: int | None = None,
    full_epochs: int = 30,
) -> dict[str, float]:
    """Ngoại suy số giờ GPU cho bản đầy đủ từ số đo của pilot (tuyến tính theo study).

    Args:
        stage_seconds: Thời gian từng bước precompute đo trên pilot.
        epoch_seconds: Thời gian trung bình một epoch của pilot (tùy chọn).
        pilot_train, full_train: Số study train của pilot / bản đầy đủ.

    Returns:
        Dict ``bước → số giờ``, kèm khóa ``"total"``.
    """
    if pilot_studies <= 0:
        raise ValueError("pilot_studies phải > 0")
    hours = {
        name: seconds / pilot_studies * full_studies / 3600
        for name, seconds in stage_seconds.items()
    }
    if epoch_seconds is not None and pilot_train and full_train:
        hours[TRAIN_KEY] = epoch_seconds / pilot_train * full_train * full_epochs / 3600
    hours["total"] = sum(hours.values())
    return hours


__all__ = ["NON_RETRYABLE", "StageTimer", "estimate_full_run"]
