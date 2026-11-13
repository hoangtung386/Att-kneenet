"""Cache volume đã tiền xử lý trên đĩa.

Mỗi lần lấy mẫu, ``KneeDataset`` phải đọc hàng chục file DICOM, resize và chuẩn
hóa. Đo trên pilot: ~0.63 s/study, gần như toàn bộ là IO và resize, lặp lại y hệt
ở mọi epoch. Volume sau ``build_volume`` chỉ phụ thuộc vào series và cấu hình dữ
liệu nên cache được **một lần** (float16, ~2 MB mỗi series ở cấu hình mặc định).

Khóa cache gồm mọi tham số làm đổi nội dung volume (shape, chế độ chuẩn hóa) và
việc series có mask hay không. Nếu bạn *sinh lại* mask cho series đã có cache,
hãy xóa thư mục ``artifacts/volumes`` hoặc dùng ``overwrite``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from knee_mri.config.schema import DataCfg
from knee_mri.data.volume import build_volume
from knee_mri.utils.io import save_npy


def cache_key(cfg: DataCfg, *, has_mask: bool) -> str:
    """Chuỗi ngắn định danh cấu hình dựng volume, dùng làm tên thư mục con."""
    parts = (cfg.target_shape, cfg.norm_mode, has_mask)
    digest = hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:10]
    return f"{'x'.join(map(str, cfg.target_shape))}-{digest}"


def cache_path(
    root: Path, cfg: DataCfg, study_uid: str, series_uid: str, *, has_mask: bool
) -> Path:
    """Đường dẫn file cache của một series."""
    return root / cache_key(cfg, has_mask=has_mask) / study_uid / f"{series_uid}.npy"


def build_volume_cached(
    series_dir: Path,
    cfg: DataCfg,
    *,
    cache_root: Path,
    study_uid: str,
    series_uid: str,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Như :func:`~knee_mri.data.volume.build_volume` nhưng có cache đĩa.

    Lỗi dựng volume (series rỗng, shape hỏng) được truyền nguyên vẹn lên gọi và
    **không** được cache.
    """
    path = cache_path(cache_root, cfg, study_uid, series_uid, has_mask=mask is not None)
    if path.is_file():
        try:
            return np.load(path).astype(np.float32)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)  # file ghi dở do runtime bị ngắt

    volume = build_volume(series_dir, cfg, mask=mask)
    stored = volume.astype(np.float16)
    save_npy(stored, path)
    return stored.astype(np.float32)  # lần đầu và các lần sau cho cùng một giá trị


__all__ = ["build_volume_cached", "cache_key", "cache_path"]
