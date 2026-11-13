"""Lưu/tải mask ROI và cắt volume theo bounding box của mask."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from knee_mri.utils.io import load_npy, save_npy


def mask_path(mask_dir: Path, study_uid: str, series_uid: str) -> Path:
    """Đường dẫn chuẩn của mask: ``<mask_dir>/<study>/<series>.npy``."""
    return mask_dir / study_uid / f"{series_uid}.npy"


def save_mask(mask: np.ndarray, study_uid: str, series_uid: str, mask_dir: Path) -> Path:
    """Ghi mask ``(D, H, W)`` kiểu uint8 (giá trị 0/1)."""
    return save_npy(mask.astype(np.uint8), mask_path(mask_dir, study_uid, series_uid))


def load_mask(study_uid: str, series_uid: str, mask_dir: Path) -> np.ndarray | None:
    """Đọc mask, trả về ``None`` nếu chưa được sinh."""
    path = mask_path(mask_dir, study_uid, series_uid)
    if not path.is_file():
        return None
    return load_npy(path, dtype=np.uint8)


def center_box_mask(shape: tuple[int, int, int], fraction: float = 0.5) -> np.ndarray:
    """Mask fallback: hộp ở giữa mỗi slice, chiếm ``fraction`` chiều H và W.

    Dùng khi SAM/VLM chưa sẵn sàng, để pipeline vẫn chạy được đầu-cuối.
    """
    depth, height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    half_h = int(height * fraction / 2)
    half_w = int(width * fraction / 2)
    mask[
        :,
        height // 2 - half_h : height // 2 + half_h,
        width // 2 - half_w : width // 2 + half_w,
    ] = 1
    return mask


def bounding_box(mask: np.ndarray, margin: int = 4) -> tuple[slice, slice, slice] | None:
    """Bounding box 3D của mask, nới thêm ``margin`` voxel mỗi phía.

    Trả về ``None`` nếu mask rỗng.
    """
    if mask is None or mask.sum() == 0:
        return None
    coords = np.nonzero(mask > 0)
    slices = []
    for axis, axis_coords in enumerate(coords):
        low = max(0, int(axis_coords.min()) - margin)
        high = min(mask.shape[axis], int(axis_coords.max()) + margin + 1)
        slices.append(slice(low, high))
    return slices[0], slices[1], slices[2]


def scale_box(
    box: tuple[slice, slice, slice],
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> tuple[slice, slice, slice]:
    """Đổi bounding box từ lưới ``from_shape`` sang lưới ``to_shape``.

    Cận dưới làm tròn xuống và cận trên làm tròn lên, nên hộp sau khi đổi luôn
    chứa trọn vùng gốc và không bao giờ rỗng.
    """
    scaled: list[slice] = []
    for axis_slice, source, target in zip(box, from_shape, to_shape, strict=True):
        ratio = target / source
        low = max(0, math.floor(axis_slice.start * ratio))
        high = min(target, max(low + 1, math.ceil(axis_slice.stop * ratio)))
        scaled.append(slice(low, high))
    return scaled[0], scaled[1], scaled[2]


def roi_crop(volume: np.ndarray, mask: np.ndarray | None = None, margin: int = 4) -> np.ndarray:
    """Cắt volume về vùng ROI của mask.

    Mask có thể ở độ phân giải **khác** volume. Mask từ bước S1 được sinh trên
    volume đã resize về ``target_shape``, trong khi lúc huấn luyện ta cắt trên
    DICOM thô (hàng trăm pixel mỗi chiều). Bản cũ trả volume nguyên vẹn mỗi khi
    hai shape lệch nhau, nên mask không bao giờ có tác dụng với student. Nay
    bounding box tính trên lưới của mask rồi được quy đổi sang lưới của volume.

    Trả về volume nguyên vẹn nếu mask là ``None``, rỗng, hoặc không phải mảng 3
    chiều. Kết quả có shape **thay đổi** — hàm gọi phải resize lại về
    :attr:`~knee_mri.config.schema.DataCfg.target_shape` (xem
    :func:`knee_mri.data.volume.build_volume`).

    Args:
        margin: Số voxel nới thêm quanh hộp, tính theo lưới của **mask**.
    """
    if mask is None or mask.ndim != 3 or volume.ndim != 3:
        return volume
    box = bounding_box(mask, margin)
    if box is None:
        return volume
    if mask.shape != volume.shape:
        box = scale_box(box, mask.shape, volume.shape)
    return volume[box]


__all__ = [
    "bounding_box",
    "center_box_mask",
    "load_mask",
    "mask_path",
    "roi_crop",
    "save_mask",
    "scale_box",
]
