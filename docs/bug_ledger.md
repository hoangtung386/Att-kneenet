# Sổ tra cứu mã lỗi (P0–P3)

Docstring và test trong repo dẫn các mã như `P0-6` hay `P2-5`. Đó là mã của những lỗi
đã sửa trong đợt refactor đầu tiên. Tài liệu gốc ghi các mã này (`refactor_guide.md`)
**không còn trong repo**, nên bảng dưới được dựng lại từ chính các docstring và test đã
dẫn mã. Nó chỉ mô tả đúng những gì code và test còn ghi lại, không thêm chi tiết nào khác.

Mức độ theo tiền tố: **P0** làm sai kết quả hoặc làm pipeline vỡ, **P1** làm chậm hoặc
thiếu chức năng thiết yếu, **P2** sai về dữ liệu/nhãn, **P3** vệ sinh code.

| Mã | Lỗi cũ | Nơi đã sửa | Test khóa lại |
|---|---|---|---|
| P0-1 | Test import `METADATA_DIM` không tồn tại trong config, nên chưa từng chạy được | `constants.py` | `test_pipeline_smoke.py` |
| P0-3 | Móc attention bằng forward hook trên `nn.TransformerEncoderLayer`, danh sách luôn rỗng | `models/student/blocks.py` | `test_student.py::TestAttention` |
| P0-4 | Hệ quả của P0-3: biến `cam` chưa gán gây `UnboundLocalError`, CAM là code chết | `explain/cam.py` | `test_student.py::TestCamPipeline` |
| P0-5 | `nn.Parameter` `pos_embed` tạo lại trong `forward`, nằm ngoài optimizer nên luôn bằng 0 | `models/student/vit3d.py` | `test_student.py::TestPositionEmbedding` |
| P0-6 | `roi_crop` trả shape tùy ý, `torch.cat` vỡ khi H/W lệch | `data/volume.py`, `data/collate.py` | `test_data.py::TestVolumeShapeInvariant` |
| P0-7 | Asymmetric Loss cài ngược hướng (mẫu âm dễ nhận trọng số ≈ 1), phần dương sai công thức | `training/losses.py` | `test_losses.py::TestAsymmetricLoss` |
| P0-8 | `auc_margin_loss` trả `float` khi không có cặp dương/âm, cắt nhánh gradient | `training/losses.py` | `test_losses.py::TestAucMarginLoss` |
| P0-9 | Cấu hình là hằng số cấp module, ghi đè từ notebook bị nuốt im lặng | `config/schema.py` | `test_config.py::TestOverridesActuallyApply` |
| P1-1 | `read_csv` gọi lại trong `select_series` ở mỗi sample, mỗi epoch | `data/catalog.py` | `test_data.py::TestCatalog::test_lookups_do_not_touch_disk` |
| P1-2 | Tương tự P1-1 với `load_study_report` | `data/catalog.py` | `test_data.py::TestCatalog::test_lookups_do_not_touch_disk` |
| P1-3 | Khởi tạo một VLM cho mỗi sample trong `Dataset`, OOM khi `num_workers > 0` | `pipeline/weak_labels.py` | |
| P1-4 | Lấy `guidance_dim` bằng `next(iter(loader))`, tiêu tốn một batch để đọc một số | `training/trainer.py` | |
| P1-5 | `num_workers` cố định bằng 0 | `training/trainer.py` | |
| P1-6 | Resize đi qua `uint8` (chỉ 256 mức) làm mất dải động MRI | `data/normalize.py` | `test_normalize.py` |
| P1-7 | Không có validation, 58 study gold không được dùng, không biết checkpoint nào tốt | `data/splits.py`, `training/checkpoint.py` | `test_data.py::TestSplits` |
| P1-8 | Thiếu mixed precision, gradient clipping, LR scheduler; `torch.load` không `weights_only` | `training/trainer.py`, `training/checkpoint.py` | `test_pipeline_smoke.py::TestCheckpointRoundTrip` |
| P2-1 | `Medial` và `Lateral Meniscus` dùng chung từ khóa `"menisc"` nên luôn bằng nhau (58/58) | `labeling/keywords.py` | `test_labeling.py::TestMeniscusLaterality` |
| P2-2 | `"lcl"` (dây chằng ngoài) bị xếp vào nhãn `MCL` (dây chằng trong) | `labeling/keywords.py` | `test_labeling.py::TestCollateralLigament` |
| P2-3 | Phủ định chỉ nhận tiếng Anh trong khi report đa ngôn ngữ | `labeling/keywords.py`, `labeling/rule_based.py` | `test_labeling.py::TestMultilingualNegation` |
| P2-4 | Ảnh `(H, W)` float trong `[-1, 1]` đưa thẳng cho SAM thay vì RGB uint8 | `models/sam_masker.py`, `data/normalize.py` | `test_normalize.py` |
| P2-5 | `(volume * 255).astype(uint8)` trên `[-1, 1]` làm giá trị âm wrap-around thành nhiễu | `models/teachers/gemma_guidance.py`, `data/normalize.py` | `test_normalize.py`, `test_model_wrappers.py` |
| P2-7 | File guidance lệch chiều với `guidance_dim` chỉ vỡ giữa lúc huấn luyện | `models/teachers/gemma_guidance.py` | |
| P3-2 | Import module tạo thư mục artifacts rác | `config/schema.py` (`Paths.ensure`) | `test_config.py::TestPaths` |
| P3-6 | Dựng volume rồi mới kiểm tra mask đã tồn tại, lãng phí IO | `pipeline/masks.py` | |
| P3-8 | Hai bản cài đặt chuẩn hóa song song bị lệch nhau | `data/normalize.py` | |
| P3-10 | `print()` rải rác thay vì logging có timestamp | `utils/logging.py` | |

Các mã P0-2, P2-6, P3-1, P3-3 đến P3-5, P3-7 và P3-9 không còn được nhắc ở đâu trong repo,
nên không có thông tin để ghi.

## Những lỗi tìm ra sau đó

Không dùng mã P. Xem [root_cause_analysis.md](root_cause_analysis.md) cho phân tích và
trạng thái từng mục.
