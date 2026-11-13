"""Đọc DICOM series thành mảng numpy thô — **không** chuẩn hóa, **không** resize.

Tách riêng khỏi :mod:`knee_mri.data.volume` để mỗi module một trách nhiệm: ở đây
chỉ có chuyện đọc file và sắp xếp lát cắt đúng thứ tự giải phẫu.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from knee_mri.constants import DICOM_SUFFIX
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


class EmptySeriesError(RuntimeError):
    """Thư mục series không chứa file DICOM đọc được nào."""


def list_dicom_files(series_dir: Path) -> list[Path]:
    """Liệt kê file DICOM trong thư mục, đã sắp theo tên để thứ tự ổn định."""
    return sorted(p for p in series_dir.iterdir() if p.suffix.lower() == DICOM_SUFFIX)


def _slice_normal(dataset: FileDataset) -> np.ndarray | None:
    """Pháp tuyến của mặt phẳng lát cắt, hướng về phía dương của trục trội.

    Tích có hướng của hai vector hướng hàng/cột (``ImageOrientationPatient``) cho
    trục mà các lát xếp dọc theo: trục z với axial, x với sagittal, y với
    coronal. Dấu của pháp tuyến được cố định sao cho thành phần lớn nhất luôn
    dương, nên cùng một mặt phẳng giải phẫu luôn được sắp cùng chiều bất kể máy
    chụp ghi hướng hàng/cột theo quy ước nào.
    """
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if orientation is None:
        return None
    try:
        values = np.asarray([float(v) for v in orientation], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape != (6,):
        return None
    normal = np.cross(values[:3], values[3:])
    if not np.any(normal):
        return None
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return normal


def _slice_sort_key(dataset: FileDataset) -> tuple[float, float]:
    """Khóa sắp xếp: tọa độ dọc pháp tuyến lát, dự phòng ``InstanceNumber``.

    Bản cũ chỉ dùng ``ImagePositionPatient[2]`` (trục z). Với series sagittal và
    coronal, các lát trải dọc trục x/y nên z gần như hằng, thứ tự phụ thuộc vào
    độ nghiêng ngẫu nhiên của máy và chiều lát bị đảo tùy series. Chiếu vị trí
    lên pháp tuyến cho thứ tự đúng và nhất quán cho cả ba mặt phẳng.
    """
    position = getattr(dataset, "ImagePositionPatient", None)
    instance = getattr(dataset, "InstanceNumber", None)
    primary = float("inf")
    if position is not None:
        try:
            coords = np.asarray([float(v) for v in position], dtype=np.float64)
            normal = _slice_normal(dataset)
            primary = float(coords @ normal) if normal is not None else float(coords[2])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        secondary = float(instance) if instance is not None else 0.0
    except (TypeError, ValueError):
        secondary = 0.0
    return primary, secondary


def slice_pixels(dataset: FileDataset) -> np.ndarray:
    """Lấy mảng pixel của một lát cắt ở đơn vị gốc của máy chụp.

    Cố ý **không** áp VOI LUT / window-center. Thẻ ``WindowCenter`` được máy ghi
    riêng cho từng lát (đo trên dữ liệu thật: 28–63 giá trị khác nhau trong một
    series), nên áp theo lát làm cùng một mô bị nhân hệ số khác nhau 25–80% giữa
    các lát kề nhau. Chuẩn hóa percentile theo volume ở bước sau không sửa được
    sai lệch *giữa* các lát. ``RescaleSlope``/``RescaleIntercept`` là hằng số
    của cả series nên vẫn được áp.
    """
    array = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        array = array * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array  # MONOCHROME1: giá trị cao = tối
    return array


def read_series(series_dir: Path, *, max_slices: int | None = None) -> np.ndarray:
    """Đọc toàn bộ series thành mảng ``(D, H, W)`` float32 ở đơn vị gốc.

    Args:
        series_dir: Thư mục chứa các file ``.dcm`` của một series.
        max_slices: Nếu có, chỉ giữ ``max_slices`` lát ở **giữa** chuỗi. Vùng
            giữa chứa phần lớn giải phẫu đáng quan tâm của đầu gối, còn hai đầu
            thường là không khí và mô mềm ngoại vi.

    Raises:
        EmptySeriesError: khi không đọc được lát nào.
    """
    files = list_dicom_files(series_dir)
    if not files:
        raise EmptySeriesError(f"Không có file DICOM nào trong {series_dir}")

    datasets: list[FileDataset] = []
    for path in files:
        try:
            datasets.append(pydicom.dcmread(path))
        except Exception as exc:  # noqa: BLE001 - file hỏng thì bỏ qua, không dừng cả series
            logger.warning("Bỏ qua file DICOM lỗi %s: %s", path.name, exc)

    if not datasets:
        raise EmptySeriesError(f"Không đọc được file DICOM nào trong {series_dir}")

    datasets.sort(key=_slice_sort_key)
    if max_slices is not None and len(datasets) > max_slices:
        start = (len(datasets) - max_slices) // 2
        datasets = datasets[start : start + max_slices]

    planes = [slice_pixels(ds) for ds in datasets]
    shapes = {plane.shape for plane in planes}
    if len(shapes) > 1:
        # Series trộn nhiều độ phân giải: giữ nhóm shape phổ biến nhất.
        dominant = max(shapes, key=lambda s: sum(p.shape == s for p in planes))
        logger.warning(
            "Series %s có %d shape khác nhau, giữ lại %s.", series_dir.name, len(shapes), dominant
        )
        planes = [plane for plane in planes if plane.shape == dominant]

    return np.stack(planes, axis=0).astype(np.float32)


__all__ = ["EmptySeriesError", "list_dicom_files", "read_series", "slice_pixels"]
