"""Augmentation cho volume đã chuẩn hóa, chỉ dùng numpy.

Bản cũ không có augmentation nào, nên một ViT-3D 35M tham số học thuộc vài
nghìn volume ngay từ những epoch đầu. Ở đây chỉ chọn các phép biến đổi **bảo
toàn nhãn một cách an toàn** cho MRI gối:

* thay đổi tương phản/độ sáng — mô phỏng sự khác biệt giữa các máy và chuỗi xung;
* nhiễu Gaussian nhẹ — mô phỏng nhiễu thu nhận;
* tịnh tiến nhỏ — vị trí gối trong khung hình không cố định.

Cố ý **không lật** theo trục nào: lật trong mặt phẳng sagittal đổi trước/sau
(ACL nằm ở vị trí đặc trưng), lật theo trục D đổi trong/ngoài ở series
coronal/axial, tức đổi nghĩa của nhãn ``Medial *`` / ``Lateral *``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Dải giá trị của volume sau ``volume_percentile_pm1``.
_LOW: float = -1.0
_HIGH: float = 1.0


@dataclass(frozen=True, slots=True)
class AugmentCfg:
    """Cường độ augmentation; đặt về 0 để tắt từng phép."""

    contrast: float = 0.15  # hệ số nhân ~ U(1 - c, 1 + c)
    brightness: float = 0.10  # độ lệch cộng ~ U(-b, b)
    noise_std: float = 0.02
    max_shift: int = 4  # số voxel tịnh tiến tối đa theo H và W


def _shift_plane(volume: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Tịnh tiến theo H, W; vùng lộ ra được lấp bằng giá trị nền (nhỏ nhất)."""
    if dy == 0 and dx == 0:
        return volume
    fill = float(volume.min())
    result = np.full_like(volume, fill)
    height, width = volume.shape[1:]
    src_y = slice(max(0, -dy), min(height, height - dy))
    dst_y = slice(max(0, dy), min(height, height + dy))
    src_x = slice(max(0, -dx), min(width, width - dx))
    dst_x = slice(max(0, dx), min(width, width + dx))
    result[:, dst_y, dst_x] = volume[:, src_y, src_x]
    return result


def augment_volume(
    volume: np.ndarray, rng: np.random.Generator, cfg: AugmentCfg | None = None
) -> np.ndarray:
    """Áp một lần biến đổi ngẫu nhiên lên volume ``(D, H, W)`` trong ``[-1, 1]``.

    Luôn trả mảng float32 **mới** cùng shape, không sửa mảng đầu vào.
    """
    cfg = cfg or AugmentCfg()
    result = np.asarray(volume, dtype=np.float32)

    if cfg.max_shift > 0:
        dy, dx = (int(v) for v in rng.integers(-cfg.max_shift, cfg.max_shift + 1, size=2))
        result = _shift_plane(result, dy, dx)

    scale = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
    offset = rng.uniform(-cfg.brightness, cfg.brightness)
    result = result * np.float32(scale) + np.float32(offset)

    if cfg.noise_std > 0:
        result = result + rng.normal(0.0, cfg.noise_std, size=result.shape).astype(np.float32)

    return np.clip(result, _LOW, _HIGH).astype(np.float32)


__all__ = ["AugmentCfg", "augment_volume"]
