"""Test đọc DICOM: thứ tự lát theo hình học và pixel không bị windowing theo lát."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from knee_mri.data.dicom_io import _slice_normal, _slice_sort_key, slice_pixels

AXIAL = [1, 0, 0, 0, 1, 0]
SAGITTAL = [0, 1, 0, 0, 0, -1]
CORONAL = [1, 0, 0, 0, 0, -1]


def _slice(orientation: list[float], position: list[float], instance: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        ImageOrientationPatient=orientation, ImagePositionPatient=position, InstanceNumber=instance
    )


def _order(datasets: list[SimpleNamespace]) -> list[int]:
    keys = [_slice_sort_key(ds) for ds in datasets]  # type: ignore[arg-type]
    return sorted(range(len(datasets)), key=keys.__getitem__)


class TestSliceOrder:
    def test_sagittal_slices_are_sorted_along_x_not_z(self) -> None:
        """Lỗi cũ: z gần như hằng ở sagittal nên thứ tự do nhiễu quyết định."""
        positions = [[30.0, 0, 0.4], [10.0, 0, 0.1], [20.0, 0, 0.3]]
        datasets = [_slice(SAGITTAL, p) for p in positions]
        assert _order(datasets) == [1, 2, 0]

    def test_coronal_slices_are_sorted_along_y(self) -> None:
        datasets = [_slice(CORONAL, [0, y, 0]) for y in (5.0, -5.0, 0.0)]
        assert _order(datasets) == [1, 2, 0]

    def test_axial_slices_are_sorted_along_z(self) -> None:
        datasets = [_slice(AXIAL, [0, 0, z]) for z in (9.0, 3.0, 6.0)]
        assert _order(datasets) == [1, 2, 0]

    def test_direction_does_not_depend_on_orientation_sign(self) -> None:
        """Hai máy ghi hướng hàng/cột ngược nhau vẫn phải cho cùng chiều lát."""
        flipped = [0, -1, 0, 0, 0, 1]  # pháp tuyến ngược dấu so với SAGITTAL
        a = _order([_slice(SAGITTAL, [x, 0, 0]) for x in (3.0, 1.0, 2.0)])
        b = _order([_slice(flipped, [x, 0, 0]) for x in (3.0, 1.0, 2.0)])
        assert a == b == [1, 2, 0]

    def test_falls_back_to_instance_number_without_geometry(self) -> None:
        datasets = [SimpleNamespace(InstanceNumber=n) for n in (3, 1, 2)]
        assert _order(datasets) == [1, 2, 0]

    def test_normal_points_to_positive_dominant_axis(self) -> None:
        normal = _slice_normal(_slice(SAGITTAL, [0, 0, 0]))  # type: ignore[arg-type]
        assert normal is not None and normal[np.argmax(np.abs(normal))] > 0


class TestSlicePixels:
    def test_ignores_per_slice_window_center(self) -> None:
        """Lỗi cũ: VOI LUT theo lát nhân cùng một mô với hệ số khác nhau."""
        pixels = np.full((4, 4), 100, dtype=np.uint16)
        narrow = SimpleNamespace(pixel_array=pixels, WindowCenter=50, WindowWidth=100)
        wide = SimpleNamespace(pixel_array=pixels, WindowCenter=500, WindowWidth=1000)
        assert np.array_equal(slice_pixels(narrow), slice_pixels(wide))  # type: ignore[arg-type]

    def test_applies_rescale(self) -> None:
        ds = SimpleNamespace(
            pixel_array=np.full((2, 2), 10, dtype=np.uint16), RescaleSlope=2, RescaleIntercept=5
        )
        assert np.allclose(slice_pixels(ds), 25.0)  # type: ignore[arg-type]

    def test_monochrome1_is_inverted(self) -> None:
        ds = SimpleNamespace(
            pixel_array=np.array([[0, 10]], dtype=np.uint16),
            PhotometricInterpretation="MONOCHROME1",
        )
        assert slice_pixels(ds).tolist() == [[10.0, 0.0]]  # type: ignore[arg-type]
