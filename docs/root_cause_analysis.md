# Vì sao model cho kết quả gần như ngẫu nhiên (AUC ≈ 0.5)

Tài liệu này là kết quả rà soát toàn bộ codebase (data pipeline, model, loss,
notebook, config) để trả lời câu hỏi: **tại sao submission và validation chỉ đạt
~0.5?** Mọi nhận định dưới đây đều kèm bằng chứng đo trực tiếp trên repo hoặc
trên output đã lưu trong notebook.

> **Về thời điểm của tài liệu.** Các bằng chứng, số dòng và đoạn code trích dưới đây
> mô tả repo ở commit `8bbf902`, **trước** đợt refactor lần hai. Chúng được giữ nguyên làm
> hồ sơ lịch sử. Hiện trạng từng nguyên nhân nằm ở bảng "Trạng thái sau refactor" ngay dưới.

---

## 0. Tóm tắt điều hành

Con số 0.5 không phải "model học sai", mà là **model chưa học gì cả**. Có một
nguyên nhân trực tiếp và một loạt nguyên nhân nền khiến ngay cả khi chạy đủ,
pipeline hiện tại cũng khó vượt xa mức ngẫu nhiên.

| # | Nguyên nhân | Mức độ | Bằng chứng |
|---|---|---|---|
| 1 | **Checkpoint đem đi nộp là bản pilot**: 150 study train, 3 epoch, **57 bước optimizer** cho một ViT-3D 35M tham số khởi tạo ngẫu nhiên | Quyết định | `notebooks/02_colab_train.ipynb` cell 2 & cell 21; `notebooks/04_kaggle_submit.ipynb` cell 2 |
| 2 | **Mask ROI (S1, 4 giờ SAM) không bao giờ được áp vào volume của student** — shape mask ≠ shape raw nên `roi_crop` trả nguyên volume | Nghiêm trọng | `src/knee_mri/utils/mask.py:71`, đo thực nghiệm ở §2.1 |
| 3 | **Student là ViT-3D huấn luyện từ đầu, không pretrain, không augmentation**, trên ≤4.3k volume | Nghiêm trọng | `src/knee_mri/models/student/vit3d.py`, `src/knee_mri/data/dataset.py` |
| 4 | **Ba trong bốn loss gần như không cung cấp tín hiệu**: InfoNCE với batch 8 (loss ≈ log 8 suốt pilot), AUC-margin đứng yên ở 0.99, KD khớp với teacher nhìn một volume *khác* student | Nghiêm trọng | log loss trong notebook 02, `src/knee_mri/training/trainer.py:212-230` |
| 5 | **Độ phân giải đầu vào quá thấp cho bệnh lý cần phát hiện**: 512–960 px → 128 px in-plane; 24–34 lát → nội suy lên 64 | Lớn | `configs/base.yaml`, đo DICOM ở §2.3 |
| 6 | **Tiền đề "nhãn cực thưa 1–2%" là sai**, nên ASL/AUC-margin được chọn trên giả định không đúng | Trung bình | đo trên 4407 report ở §4.3 |
| 7 | **Guidance text có khả năng cao gần giống nhau giữa các study** (mean-pool toàn bộ token, gồm ~1024 token ảnh + prompt cố định) | Trung bình (giả thuyết có cơ sở) | `src/knee_mri/models/teachers/gemma_guidance.py:107-112` |
| 8 | Tập validation = 58 study, vừa để chọn checkpoint vừa để báo cáo; nhiễu AUC ±0.1 | Trung bình | `src/knee_mri/data/splits.py`, `trainer.py:369-388` |
| 9 | VOI windowing áp **theo từng lát** với WindowCenter khác nhau → cường độ lệch 25–80% giữa các lát trong cùng series | Nhỏ–trung bình | `src/knee_mri/data/dicom_io.py:75-80`, đo ở §2.4 |

Thứ tự ưu tiên khi sửa nằm ở §6.

### Trạng thái sau refactor

| # | Nguyên nhân | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Checkpoint nộp là bản pilot | **Chưa làm** | Cần huấn luyện lại; không phải việc của code |
| 2 | Mask ROI không bao giờ được áp | **Đã sửa** | `roi_crop` quy đổi mask sang lưới của DICOM thô; `TestRoiCrop`, `TestVolumeShapeInvariant` |
| 3 | ViT-3D từ đầu, không augmentation | **Một phần** | Đã thêm augmentation, mean-pool, positional embedding 3D. Chưa có backbone pretrain |
| 4 | InfoNCE, AUC-margin, KD không có tín hiệu | **Một phần** | AUC-margin đã xóa hẳn; InfoNCE mặc định tắt và có memory bank (`train.memory_bank_size`) cho mẫu âm; KD nay khớp vùng crop với teacher. Chưa kiểm chứng bằng huấn luyện; đo cosine bằng `knee-similarity` |
| 5 | Độ phân giải đầu vào thấp | **Chưa làm** | Vẫn 128 px và 64 lát; đổi sẽ tăng bộ nhớ mạnh, cần quyết định riêng |
| 6 | Tiền đề nhãn cực thưa sai | **Đã sửa** | BCE mặc định; tài liệu đã được đính chính |
| 7 | Guidance bị pha loãng bởi token ảnh | **Đã sửa, chưa kiểm chứng** | Guidance mặc định chỉ dùng report; chưa đo cosine giữa các vector |
| 8 | Validation 58 study vừa chọn vừa báo cáo | **Chưa làm** | `extra_val_ratio` có sẵn nhưng mặc định 0 |
| 9 | VOI windowing theo lát | **Đã sửa** | `TestSlicePixels` |
| 2.5 | Hướng lát không nhất quán | **Đã sửa** | `TestSliceOrder` |
| 2.7 | Không có cache volume | **Đã sửa, mặc định tắt** | `data.cache_volumes` |

Baseline để quyết định bước tiếp theo: `knee-probe` (xem §6 mục 2).


---

## 1. Thực tế đã chạy: model gần như chưa được huấn luyện

### 1.1 Chỉ có pilot, không có full run

Notebook `02_colab_train.ipynb` (output đã lưu) cho thấy toàn bộ lần chạy là
`MODE = 'pilot'`:

```
PILOT_EXTRA = 150     # study không nhãn thêm vào
PILOT_EPOCHS = 3
Split(train=150, val=58) | gold trong val: 58
Bắt đầu huấn luyện trên cuda | 150 study train, 58 study val | 3 epoch
epoch 0 step 0/19 ...
```

Với `batch_size = 8` (env colab) → 19 bước/epoch × 3 epoch = **57 bước
optimizer**. Trong đó 5% đầu là warmup (≈3 bước). Student có 35.05M tham số
(đo bằng `build_bundle`), khởi tạo ngẫu nhiên. 57 bước không đủ để một ViT học
bất cứ gì ngoài bias của head.

Kết quả validation khớp hoàn toàn với "chưa học":

```
epoch 0: macro AUC = 0.5388
epoch 1: macro AUC = 0.5387
epoch 2: macro AUC = 0.5366
Đánh giá độc lập (Predictor, chỉ ảnh): macro AUC = 0.5372
```

Ba epoch, AUC không nhúc nhích. Nhìn theo nhãn: Effusion 0.72, MCL 0.64, ACL
0.61, còn Medial Meniscus **0.38** và Contusion **0.40** — tức *dưới* ngẫu nhiên.
Đây là dấu hiệu điển hình của xác suất ngẫu nhiên trên 58 mẫu: vài nhãn tình cờ
trên 0.5, vài nhãn tình cờ dưới.

### 1.2 Submission dùng đúng checkpoint pilot này

`04_kaggle_submit.ipynb` nạp `student_best.pt` từ Kaggle model
`hoangtung719/att-v1`. Commit `70c3a7a Submission Kaggle` nằm ngay sau commit
sửa notebook pilot, và cell 26 của notebook 02 ghi rõ ở chế độ pilot thì
**không sinh submission** — nghĩa là checkpoint `att-v1` được tải lên tay từ
`artifacts/pilot/checkpoints/student_best.pt` (epoch 1, best = 0.5388). Điểm
leaderboard ≈ 0.5 là hệ quả tất yếu.

### 1.3 Vì sao chưa chạy full: ước lượng 128 giờ A100

Cell 28 ngoại suy:

```
masks          ≈  84.8 giờ
weak_labels    ≈  14.5 giờ
teacher_feats  ≈   2.8 giờ
guidance       ≈   1.9 giờ
train          ≈  23.8 giờ
Tổng           ≈ 127.8 giờ GPU A100
```

85 giờ trong số đó là bước SAM — và như §2.1 chứng minh, bước này **không có
tác dụng gì** với student. Chi phí pipeline bị đội lên bởi một bước vô hiệu,
và đó là lý do thực tế khiến bản full chưa bao giờ được chạy.

---

## 2. Lỗi trong data pipeline

### 2.1 Mask ROI không bao giờ được áp (lỗi im lặng)

Luồng S1 (`src/knee_mri/pipeline/masks.py:49-56`):

```python
volume = build_volume(series_dir, cfg.data)      # → shape (64, 128, 128)
mask = masker.generate_3d(volume, bbox=...)      # → shape (64, 128, 128)
save_mask(mask, ...)
```

Mask được sinh và lưu ở **target_shape** `(64, 128, 128)`.

Luồng huấn luyện (`src/knee_mri/data/volume.py:57-59`):

```python
raw = read_series(series_dir, max_slices=...)    # → shape (34, 960, 960) — raw DICOM
if mask is not None:
    raw = roi_crop(raw, mask, margin=roi_margin)
```

`roi_crop` (`src/knee_mri/utils/mask.py:71`):

```python
if mask is None or mask.shape != volume.shape:
    return volume          # ← (64,128,128) != (34,960,960) → trả nguyên raw
```

Đo thực nghiệm trên series test local:

```
raw shape (34, 960, 960) | built volume (64, 128, 128) | mask (64, 128, 128)
roi_crop(raw, mask) returns same object as raw? -> True
build_volume with mask identical to without mask? -> True
```

Hệ quả:

- Student, `Predictor`, `compute_cam` đều nhìn **toàn bộ FOV** chưa crop, dù
  docs (`docs/normalization.md` §"Thứ tự: crop ROI → resize → chuẩn hóa") và
  README mô tả ngược lại.
- Test `tests/test_data.py:92 test_mismatched_mask_is_ignored_safely` khóa
  hành vi "bỏ qua im lặng" này như một tính năng, nên lỗi không bao giờ nổ.
- Teacher 3DINO thì **có** crop: `pipeline/features.py:49-54` gọi
  `teacher.extract(volume, mask)` với `volume` đã là `(64,128,128)` nên shape
  khớp và `roi_crop` hoạt động. Vậy **teacher nhìn vùng trung tâm ~50%, student
  nhìn toàn ảnh**, rồi KD ép student khớp đặc trưng của một ảnh khác. Xem §4.4.
- IoU CAM/ROI = 0.089 trong notebook không nói lên gì: CAM được dựng trên
  volume chưa crop và so với mask được resize về cùng shape.

### 2.2 Bản thân mask SAM cũng không phải "ROI bệnh lý"

`SamMaskGenerator.generate_3d` (`src/knee_mri/models/sam_masker.py:77-96`):
prompt là **hộp trung tâm 50%** của lát giữa, chạy SAM một lần, rồi
`np.broadcast_to` sang mọi lát. Không có VLM sinh bbox như sơ đồ kiến trúc
mô tả. Với MRI gối, SAM + box prompt trả về một vùng tùy ý bên trong hộp
(thường là toàn hộp hoặc một xương). Kể cả khi crop có tác dụng, nó chỉ là
center-crop 50% — không mang thông tin bệnh lý.

### 2.3 Độ phân giải đầu vào quá thấp so với bệnh lý

DICOM thật (đo trên 15 series test local): in-plane **512–960 px**, pixel
spacing 0.16–0.35 mm, **24–40 lát** dày 3–4 mm (một series 3D có 160 lát).

Config: `volume_size: [128, 128]`, `max_slices: 64`, `patch_size: [8, 16, 16]`.

- In-plane giảm 4–7.5 lần → pixel hiệu dụng ≈ 1.2–2.3 mm. Rách sụn chêm, đứt
  ACL một phần, khuyết sụn độ 1–2 là cấu trúc cỡ 1–3 mm. Tín hiệu bị xóa
  trước khi vào model.
- Trục D: 24–34 lát được **nội suy lên 64** (`_resize_axis` lặp dữ liệu), rồi
  patch D = 8 gộp 8 "lát" trong đó chỉ có ~3–4 lát thật. Tốn compute, không
  thêm thông tin.
- Mỗi patch 8×16×16 → chỉ **512 token** cho cả khối gối. ViT phải ra quyết
  định ở granularity rất thô.

### 2.4 VOI windowing theo từng lát làm lệch cường độ trong cùng series

`slice_pixels` (`src/knee_mri/data/dicom_io.py:75-80`) gọi `apply_voi_lut`
cho **từng lát** nếu lát đó có `WindowCenter`. Đo trên dữ liệu thật:

```
Axial    n=34  : 29 WindowCenter khác nhau (1049–1233), tỉ lệ windowed/raw theo lát 1.63 → 2.01
Sagittal n=30  : 28 WindowCenter khác nhau (795–1005),  tỉ lệ 1.96 → 2.52
Axial 3D n=160 : 63 WindowCenter khác nhau (136–216),   tỉ lệ 17.2 → 32.2
```

Cùng một mô, cùng series, bị nhân với hệ số khác nhau 25–80% tùy lát (gần 2×
ở series 3D). Sau đó `volume_percentile` chuẩn hóa theo **toàn volume** nên
không sửa được lệch *giữa các lát*. `docs/normalization.md` coi "flicker giữa
lát" là tín hiệu 3D thật cần giữ; thực ra một phần đáng kể là artifact
windowing do chính code tạo ra.

### 2.5 Thứ tự và hướng lát không nhất quán

`_slice_sort_key` (`dicom_io.py:38-48`) sắp theo `ImagePositionPatient[2]`
(trục z). Với series **sagittal**, lát di chuyển theo x, z gần như hằng
(ptp z = 0–4 mm, có series chỉ 4 giá trị z duy nhất cho 24 lát). Trên dữ liệu
local, thứ tự vẫn đơn điệu nhờ tilt nhỏ hoặc fallback InstanceNumber, nhưng
**chiều** thì đảo tùy máy: cùng là sagittal, có series ra `1…24`, series khác
ra `24…1`. Model không có augmentation lật nên phải tự học cả hai chiều.

### 2.6 Trộn ba mặt phẳng vào cùng một ViT-3D

`select_series` lấy top-4 series theo điểm (sagittal ×3, fluid ×2, fat ×1).
Trên dữ liệu thật mỗi study có 5–6 series ở cả 3 mặt phẳng, nên batch luôn
trộn volume mà trục D là *x*, *y*, hoặc *z* giải phẫu. Student dùng chung
`patch_embed` và `pos_embed` cho cả ba, chỉ có 5 chiều metadata ở head để
phân biệt. Với model huấn luyện từ đầu trên vài nghìn mẫu, đây là một bài toán
khó thêm vào không cần thiết.

### 2.7 Không có augmentation, không cache volume

`KneeDataset.__getitem__` đọc DICOM → resize → normalize mỗi lần lấy mẫu, không
có bất kỳ augmentation nào (grep `flip|augment|RandomCrop` trong `src/` trả
rỗng). Pilot: 95 s/epoch cho 150 study ≈ 0.63 s/study, toàn bộ là IO + resize
từ Google Drive. Bản full ≈ 45 phút/epoch chỉ để đọc lại dữ liệu y hệt.

---

## 3. Kiến trúc model

### 3.1 ViT-3D 35M tham số huấn luyện từ đầu

`ViT3DStudent`: 6 khối, dim 768, 8 head, MLP 2048, khởi tạo `trunc_normal`.
Không có pretrained weight nào. ViT nổi tiếng là **đói dữ liệu**: ViT-B từ
đầu trên ImageNet-1k (1.3M ảnh) còn thua ResNet; ở đây là ≤4349 volume 3D.
Trong khi đó pipeline đã có sẵn **3DINO-ViT (ViT-L, pretrain 100k scan y tế)**
nhưng chỉ dùng nó làm mục tiêu KD, không dùng làm backbone.

Cách rẻ và gần như chắc chắn tốt hơn: linear probe / MLP trên đặc trưng 3DINO
đã cache (1024 chiều/series). Cache này **đã có sẵn** sau bước
`precompute_teacher_features`, huấn luyện trong vài phút trên CPU.

### 3.2 Positional embedding 1D cho lưới 3D

`sinusoidal_position_embedding(num_patches + 1, dim)` (`blocks.py:77-90`) đánh
số 512 patch theo một chỉ số phẳng. Hai patch kề nhau theo H nhưng cách nhau
theo W bị coi là xa như hai patch ở hai đầu volume. Học được, nhưng tốn thêm
dữ liệu mà ta không có.

### 3.3 Head chỉ dùng CLS token

`self.norm(tokens[:, 0])`. Với model chưa pretrain, CLS token ở 57 bước (hay cả
vài nghìn bước) gần như không mang thông tin không gian; mean-pool patch token
thường ổn định hơn ở giai đoạn đầu.

---

## 4. Loss và tín hiệu giám sát

Tổng loss: `L = 1.0·KD + 1.0·InfoNCE + 1.0·ASL + 0.5·AUC`. Log pilot:

```
epoch 0: kd=0.7081 contrast=2.0634 cls=1.3533 auc=0.9966
epoch 1: kd=0.5502 contrast=1.9680 cls=1.2938 auc=0.9971
epoch 2: kd=0.5405 contrast=1.8844 cls=1.2581 auc=0.9915
```

### 4.1 InfoNCE với batch 8 ≈ không học

`info_nce` (`losses.py:25-44`): với B study trong batch, loss ngẫu nhiên là
`log(B) = log 8 = 2.079`. Giá trị quan sát 2.06 → 1.88: gần như sàn ngẫu nhiên
suốt pilot, và là **thành phần lớn nhất** của tổng loss (≈45%). CLIP-style
contrastive cần batch hàng trăm–nghìn; docstring của chính hàm này cũng cảnh
báo điều đó. Với `batch_size: 8` và không có memory bank, nhánh "text-guided"
— ý tưởng cốt lõi của toàn bộ kiến trúc — thực tế không truyền được tri thức.

### 4.2 Vector guidance gần như giống nhau giữa các study (giả thuyết)

`GemmaGuidanceEncoder.encode` (`gemma_guidance.py:107-112`):

```python
hidden = outputs.hidden_states[-1]
pooled = (hidden * attention_mask).sum(1) / attention_mask.sum(1)
```

Mean-pool **toàn bộ** chuỗi: prompt cố định (~60 token) + **4 ảnh × 256 soft
token của Gemma 3 = 1024 token ảnh** + report (~200–400 token). Khoảng 70–80%
vector là trung bình của token ảnh (4 lát giữa, rất giống nhau giữa các gối)
và prompt giống hệt nhau. Phần report — thứ duy nhất chứa tri thức lâm sàng
riêng của study — bị pha loãng. Thêm nữa, hidden state cuối của decoder LLM
có các chiều "massive activation" lấn át cosine similarity.

Chưa đo trực tiếp vì cache guidance không có ở local, nhưng có thể kiểm tra
nhanh: cosine giữa các vector `artifacts/pilot/guidance/*.npy` — nếu trung bình
> 0.95 thì giả thuyết đúng. Projector `text_contrast` (16.7M tham số, 3840→3840→512)
trên 150 mẫu cũng chỉ có thể ghi nhớ.

### 4.3 Tiền đề "nhãn cực thưa" sai → chọn ASL và AUC-margin trên giả định không đúng

`docs/data_notes.md` và docstring của `asymmetric_loss` nói "dương tính ~1–2%
mỗi nhãn". Đo thật:

| Nhãn | Weak label (luật, 4407 study) | Gold (58 study) |
|---|---|---|
| ACL | 26% | 41% |
| MCL | 22% | 16% |
| Medial Meniscus | 39% | 45% |
| Lateral Meniscus | 28% | 40% |
| Medial OA | 16% | 26% |
| Lateral OA | 13% | 19% |
| PF OA | 17% | 36% |
| Effusion | 36% | 60% |
| Synovitis | 10% | 47% |
| Baker's | 20% | 21% |
| Contusion | 17% | 33% |
| Fracture | 8% | 31% |

Trung bình 2.5 nhãn dương/study; 31% study không có nhãn dương nào. Đây là
bài toán multi-label **cân bằng vừa phải**, không phải cực thưa. Hệ quả:

- ASL với `gamma_neg = 4`, `clip = 0.05` hạ trọng số mẫu âm rất mạnh — thiết
  kế cho tỉ lệ dương ~1% (ImageNet-21k multi-label). Ở tỉ lệ 20–40%, nó làm
  model thiên về dương và làm méo xác suất. BCE thường (có thể kèm `pos_weight`)
  là lựa chọn mặc định đúng.
- Phân bố weak label (luật/VLM trên 4407) và gold (58) **lệch nhau đáng kể**
  (Synovitis 10% vs 47%, Fracture 8% vs 31%): 58 study gold không phải mẫu
  ngẫu nhiên của tập train; chúng nặng bệnh hơn. Validation trên 58 study này
  đo một phân bố khác với phân bố huấn luyện.

### 4.4 AUC-margin loss đứng yên

> **Đã xử lý:** hàm `auc_margin_loss` và các tham số `train.loss.auc`, `train.auc_margin` đã bị xóa khỏi codebase.

`auc_margin_loss` với `margin = 1.0`: khi logit đều gần 0 (model mới khởi
tạo), `relu(1 − (pos − neg)) ≈ 1` cho mọi cặp → loss ≈ 1.0 và gradient hằng.
Log: 0.9966 → 0.9971 → 0.9915. Loss này chỉ có nghĩa khi logit đã tách; ở giai
đoạn này nó chỉ thêm nhiễu.

### 4.5 KD khớp với teacher nhìn một volume khác

Như §2.1: teacher 3DINO nhận volume **đã center-crop 50%** rồi resize 112³;
student nhận **toàn FOV** 64×128×128. `kd_cosine_loss` ép `proj(student_cls)`
cùng hướng với đặc trưng của một vùng ảnh student không được thấy đầy đủ.
KD giảm nhanh 0.97 → 0.54 không chứng minh student học được gì: đặc trưng CLS
của 3DINO trên các gối MRI có cosine similarity rất cao với nhau, nên
`image_projector` (MLP 2 lớp) chỉ cần xuất ra **vector trung bình** là đã đạt
cosine ≈ 0.5–0.6. Có thể kiểm chứng bằng cosine trung bình giữa các file
`artifacts/pilot/teacher_feats/**/*.npy`.

### 4.6 Nhãn yếu: trần chất lượng và quy ước "không nhắc = âm"

- VLM weak label đạt macro-F1 0.74 trên gold (precision 0.66, recall 0.87).
  Tức ~1/3 nhãn dương mà model được dạy là sai. Với 150 study pilot, đó là
  nhiễu áp đảo; với 4349 study, nó đặt trần AUC đáng kể nhưng không phải lý
  do của 0.5.
- Cả luật lẫn VLM coi "report không nhắc tới" là âm. Report tiếng Tây Ban Nha /
  Đức ngắn thường chỉ liệt kê bất thường, nên âm giả ở nhãn như Synovitis
  (10% weak vs 47% gold) là hệ thống.
- `precompute_weak_labels` lùi về luật từ khóa khi VLM lỗi mà không ghi dấu
  trong file JSON, nên không phân biệt được nhãn nào đến từ nguồn nào.

---

## 5. Đánh giá và chọn model

- 58 study gold vừa là tập **chọn checkpoint** (`trainer.py:376-388`) vừa là
  tập **báo cáo**. Với 9–35 dương/nhãn, sai số chuẩn của AUC mỗi nhãn ≈ 0.08–0.12;
  chênh lệch 0.5388 vs 0.5366 giữa các epoch là nhiễu thuần.
- Như §4.3, 58 study này lệch phân bố với tập train, nên AUC đo được cũng
  không đại diện cho leaderboard.
- `evaluate_gold` và `predict_study` lấy **mean xác suất qua 4 series** ở 3 mặt
  phẳng khác nhau. Series axial gần như không nhìn thấy ACL/sụn chêm nhưng vẫn
  đóng góp 1/4 trọng số cho nhãn đó.
- Không có TTA, không ensemble, không hiệu chỉnh ngưỡng — tất cả là thứ yếu khi
  model chưa học, nhưng sẽ quan trọng sau khi sửa phần trên.

---

## 6. Thứ tự sửa đề xuất

Xếp theo tỉ lệ **lợi ích / chi phí**:

1. **Bỏ qua S1 (SAM) cho student, hoặc sửa `build_volume` để crop trên volume
   đã resize** (đồng thời xóa test `test_mismatched_mask_is_ignored_safely`
   hoặc đổi nó thành assert lỗi to). Tiết kiệm 85 giờ GPU ngay lập tức.
2. **Baseline nhanh trên đặc trưng 3DINO đã cache**: logistic regression /
   MLP nhỏ, đầu vào = concat hoặc mean của 1024-d × 4 series + metadata, nhãn
   = weak label VLM, validate trên 58 gold. Chạy trong vài phút. Nếu AUC > 0.65,
   ta biết *dữ liệu và nhãn* đủ tốt và vấn đề nằm ở student.
3. **Thay student bằng backbone có pretrain** (fine-tune 3DINO với LoRA/partial,
   hoặc 2.5D CNN pretrain ImageNet trên lát sagittal) thay vì ViT từ đầu.
   Thêm augmentation (lật, xoay nhỏ, crop, nhiễu cường độ).
4. **Tăng độ phân giải**: giữ ≥256×256 in-plane, D = số lát thật (pad thay vì
   nội suy), cân nhắc chỉ dùng series sagittal + coronal.
5. **Loss**: BCE (+`pos_weight` theo tỉ lệ thật) thay ASL; bỏ AUC-margin hoặc
   bật sau khi warm-up; InfoNCE chỉ khi có batch ≥128 hoặc memory bank, và
   guidance phải pool **chỉ trên token report** (hoặc dùng text encoder
   riêng). Nếu không làm được thì tắt nhánh contrast, nó đang là 45% loss mà
   không đóng góp gì.
6. **Tắt VOI windowing per-slice** (`slice_pixels`): dùng pixel thô + rescale
   slope/intercept, chuẩn hóa percentile theo volume như hiện tại.
7. **Chuẩn hóa hướng lát theo `ImageOrientationPatient`** thay vì chỉ
   `position[2]`, để sagittal/coronal luôn cùng chiều.
8. **Cache volume đã tiền xử lý** (npy/npz, float16) một lần trên Drive hoặc
   disk local Colab; epoch từ 45 phút xuống vài phút, khi đó mới chạy nổi 30
   epoch trên 4349 study.
9. **Validation**: dùng K-fold trên weak label để chọn checkpoint, giữ 58 gold
   *chỉ* để báo cáo; hoặc chia 58 thành 2 nửa.

Nếu chỉ làm **mục 1, 2 và 5** (vài giờ công), đã có một con số có ý nghĩa để
so sánh và biết nên đầu tư GPU vào đâu. Chạy full pipeline hiện tại 128 giờ
rất có thể vẫn cho AUC ≈ 0.55–0.6 vì các nguyên nhân nền ở §2–§4 không mất đi
khi tăng số epoch.

---

## Phụ lục: cách tái lập các số đo trong tài liệu này

```bash
# Mask không được áp (cần dataset/test_series local)
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0,"src")
from pathlib import Path; import numpy as np, pandas as pd
from knee_mri.config import load_config
from knee_mri.data.dicom_io import read_series
from knee_mri.data.volume import build_volume
from knee_mri.utils.mask import roi_crop, center_box_mask
cfg = load_config("colab", overrides={"paths": {"data_root": "dataset", "artifacts_root": "/tmp/x"}})
r = pd.read_csv("dataset/test_series.csv").iloc[0]
d = Path("dataset/test_series") / r.StudyInstanceUID / r.SeriesInstanceUID
raw = read_series(d, max_slices=cfg.data.max_slices * 3)
vol = build_volume(d, cfg.data)
mask = center_box_mask(vol.shape, 0.5)
print(raw.shape, vol.shape, mask.shape, roi_crop(raw, mask) is raw)
print(np.array_equal(build_volume(d, cfg.data, mask=mask), vol))
EOF

# Tỉ lệ nhãn dương của weak label trên 4407 report
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0,"src")
import pandas as pd, numpy as np
from knee_mri.labeling.rule_based import keyword_labeler
from knee_mri.constants import LABELS
tr = pd.read_csv("dataset/train.csv")
rows = np.array([[keyword_labeler(r)[l] for l in LABELS] for r in tr.Report])
print(dict(zip(LABELS, rows.mean(0).round(3))))
EOF

# Số tham số và số bước optimizer của pilot
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0,"src")
from knee_mri.config import load_config
from knee_mri.training.trainer import build_bundle
cfg = load_config("colab", overrides={"paths": {"data_root": "dataset", "artifacts_root": "/tmp/x"}})
b = build_bundle(cfg)
print({n: sum(p.numel() for p in m.parameters()) for n, m in b.as_dict().items()})
print("pilot steps:", 3 * -(-150 // cfg.train.batch_size))
EOF
```
