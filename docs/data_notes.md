# Ghi chú về dataset

Số liệu dưới đây đo trực tiếp trên `dataset/` bằng `StudyCatalog` và
`knee_mri.labeling`, không phải trích từ tài liệu đề bài.

## Quy mô

| Chỉ số | Giá trị |
|---|---|
| Study (train) | 4407 |
| Series (train) | 24371 |
| Study có `Report` | 4407 (100%) |
| Study có nhãn cấu trúc | **58** (1.3%) |
| Series trung bình / study | ~5.5 |

## Phân bố nhãn

Tiền đề cũ "nhãn cực thưa, ~1–2% dương tính" là **sai**. Đo trực tiếp:

| Nhãn | Weak label (luật, 4407 study) | Gold (58 study) | Dương gold |
|---|---|---|---|
| Effusion | 36% | 60% | 35 |
| Synovitis | 10% | 47% | 27 |
| Medial Meniscus | 39% | 45% | 26 |
| ACL | 26% | 41% | 23 |
| Lateral Meniscus | 28% | 40% | 23 |
| PF OA | 17% | 36% | 21 |
| Contusion | 17% | 33% | 19 |
| Fracture | 8% | 31% | 18 |
| Medial OA | 16% | 26% | 15 |
| Baker's | 20% | 21% | 12 |
| Lateral OA | 13% | 19% | 11 |
| MCL | 22% | 16% | 9 |

Trung bình 2.5 nhãn dương mỗi study; 31% study không có nhãn dương nào.

Hệ quả thiết kế:

- Dùng **BCE** làm loss phân loại mặc định. Asymmetric Loss (`gamma_neg = 4`) thiết kế
  cho nhãn dương ~1% và hạ trọng số mẫu âm quá tay ở đây; vẫn chọn được bằng `train.cls_loss`.
- Metric là **AUC**, không phải accuracy.
- 58 study gold là **validation**, không dùng để huấn luyện (`data/splits.py`). Chúng
  nặng bệnh hơn tập train (Synovitis 47% so với 10%), nên AUC trên gold đo một phân bố
  hơi khác với phân bố huấn luyện và nhiễu mạnh (9–35 dương mỗi nhãn).

## Report đa ngôn ngữ

Ước lượng bằng từ khóa đặc trưng:

| Ngôn ngữ | Số study |
|---|---|
| Anh | ~1591 |
| Tây Ban Nha | ~701 |
| Hà Lan | ~303 |
| Đức | ~244 |
| Pháp | ~80 |
| Không xác định rõ | ~1488 |

Ví dụ cùng ý "không có tràn dịch" trong 5 ngôn ngữ:

```
en  No evidence of joint effusion.
es  Sin derrame articular.
nl  Geen vocht in het gewricht.
de  Kein Gelenkerguss.
fr  Pas de épanchement articulaire.
```

Bộ gán nhãn phải nhận ra **cả năm** là phủ định. Bản trước refactor chỉ nhận
`"no "`, `"without"`, `"negativ"` (tiếng Anh), nên gán dương tính sai cho hàng
trăm study.

## Chất lượng weak label

Đo trên 58 study có nhãn gold, so bản luật từ khóa cũ và mới:

| Chỉ số | Trước refactor | Sau refactor |
|---|---|---|
| Precision | 0.421 | **0.566** |
| Recall | 0.454 | **0.533** |
| F1 | 0.437 | **0.549** (+25.7%) |
| False positive | 150 | **98** |
| Study mà `Medial Meniscus == Lateral Meniscus` | **58/58 (100%)** | 47/58 |

Dòng cuối là bằng chứng trực tiếp cho lỗi P2-1: hai nhãn sụn chêm dùng chung từ
khóa trần `"menisc"` nên **luôn** nhận giá trị giống hệt nhau — 2/12 nhãn vô giá trị.

Tái lập số liệu này:

```bash
python scripts/precompute_weak_labels.py --env local
```

## Metadata series

`Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` có ở **cả**
`train_series.csv` lẫn `test_series.csv`. Đây là đầu vào phụ trợ hợp lệ lúc
inference, khác với report (chỉ có lúc huấn luyện).

Thứ tự ưu tiên khi chọn series (`data/series_selection.py`): Sagittal (×3) >
nhạy dịch (×2) > xóa mỡ (×1). Sagittal tốt nhất cho ACL và sụn chêm; chuỗi nhạy
dịch làm nổi tràn dịch và phù tủy xương.

## Đặc điểm DICOM

- Đuôi file: `.dcm`; mỗi series có khoảng 20–40 lát.
- Độ phân giải điển hình: 512×512, nhưng đo thực tế có 512–960 px, pixel spacing
  0.16–0.35 mm, 24–40 lát dày 3–4 mm (có series 3D 160 lát); đôi khi lẫn nhiều độ
  phân giải trong một series (`dicom_io.read_series` giữ nhóm shape phổ biến nhất).
- Sắp xếp lát theo tọa độ chiếu lên **pháp tuyến mặt phẳng lát** (tích có hướng của
  `ImageOrientationPatient`, dấu cố định theo trục trội), dự phòng bằng
  `InstanceNumber`. Không dùng riêng `ImagePositionPatient[2]`: với sagittal/coronal
  các lát trải dọc trục x/y, z gần như hằng.
- **Không** áp VOI LUT/WindowCenter: máy ghi giá trị khác nhau cho từng lát (28–63 giá
  trị trong một series), làm cùng một mô lệch 25–80% giữa các lát.
- MRI **không** có thang đo tuyệt đối — xem `docs/normalization.md`.
