"""Các hàm mục tiêu cho huấn luyện student.

Bối cảnh: chỉ 58/4407 study có nhãn do người gán, nhưng **mọi** study đều có
report. Chiến lược là bóc tri thức từ 4407 cặp (ảnh, report) thay vì chỉ dựa vào
58 nhãn. Bốn thành phần:

1. :func:`bce_loss` — phân loại 12 nhãn trên nhãn yếu. Mặc định, vì nhãn chỉ
   mất cân bằng vừa phải (xem ``docs/data_notes.md``).
2. :func:`kd_cosine_loss` — giữ đặc trưng ảnh mạnh từ 3DINO-ViT.
3. :func:`asymmetric_loss` — lựa chọn cho nhãn *thực sự* cực thưa (~1%).
4. :func:`info_nce` — căn chỉnh ảnh với guidance text. Cần nhiều mẫu âm: với
   batch nhỏ phải kèm memory bank (``training/memory_bank.py``). Mặc định tắt
   (trọng số 0).

Tổng: ``L = λ_kd·L_kd + λ_cls·L_cls + λ_ctr·L_infonce``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

_EPS: float = 1e-8


def info_nce(
    image_embedding: Tensor,
    text_embedding: Tensor,
    *,
    temperature: float = 0.07,
    negative_text: Tensor | None = None,
    negative_valid: Tensor | None = None,
) -> Tensor:
    """InfoNCE giữa đặc trưng ảnh và guidance text.

    Mỗi study trong batch là một cặp dương; các study còn lại là cặp âm. Với
    batch ``B`` thì loss của model ngẫu nhiên là ``log(B)``: ở ``B = 8`` loss
    dậm chân quanh 2.08 và không truyền được tri thức.

    Để có nhiều mẫu âm mà không tăng batch, truyền thêm ``negative_text`` (từ
    :class:`~knee_mri.training.memory_bank.GuidanceBank`). Khi đó chiều
    ảnh → text so mỗi ảnh với ``B + K`` text; chiều text → ảnh vẫn chỉ trong
    batch, vì ``K`` embedding ảnh cũ sẽ lỗi thời (student đã đổi).

    Args:
        image_embedding: ``(B, D)`` đã L2-normalize.
        text_embedding: ``(B, D)`` đã L2-normalize.
        temperature: Nhiệt độ softmax; nhỏ hơn = phân biệt gắt hơn.
        negative_text: ``(K, D)`` đã L2-normalize, không gradient, hoặc ``None``.
        negative_valid: ``(B, K)`` bool, ``False`` ở chỗ phải bỏ qua (ví dụ chính
            study đó nằm trong bank). ``None`` nghĩa là mọi chỗ hợp lệ.
    """
    has_bank = negative_text is not None and negative_text.shape[0] > 0
    if image_embedding.shape[0] < 2 and not has_bank:
        # Không có cặp âm nào; trả về 0 nhưng vẫn giữ đồ thị gradient.
        return image_embedding.sum() * 0.0
    logits = image_embedding @ text_embedding.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    if not has_bank:
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2

    assert negative_text is not None
    bank_logits = image_embedding @ negative_text.detach().t() / temperature
    if negative_valid is not None:
        bank_logits = bank_logits.masked_fill(~negative_valid, float("-inf"))
    image_to_text = F.cross_entropy(torch.cat([logits, bank_logits], dim=1), targets)
    if logits.shape[0] < 2:
        return image_to_text
    return (image_to_text + F.cross_entropy(logits.t(), targets)) / 2


def bce_loss(logits: Tensor, targets: Tensor, *, pos_weight: Tensor | None = None) -> Tensor:
    """Binary cross-entropy đa nhãn, **tổng** theo nhãn rồi trung bình theo mẫu.

    Cùng quy ước rút gọn với :func:`asymmetric_loss` nên đổi qua lại giữa hai
    loss không làm đổi thang đo so với các thành phần khác của tổng loss.

    Args:
        logits: ``(B, C)`` logit thô, chưa qua sigmoid.
        targets: ``(B, C)`` giá trị trong ``{0, 1}``.
        pos_weight: ``(C,)`` hệ số nhân cho mẫu dương của từng nhãn, hoặc ``None``.
    """
    per_element = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none", pos_weight=pos_weight
    )
    return per_element.sum(dim=1).mean()


def asymmetric_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> Tensor:
    """Asymmetric Loss (Ben-Baruch et al., 2021) cho multi-label mất cân bằng.

    Thiết kế cho nhãn **cực** thưa (dương tính ~1%). Dữ liệu này có 8–39% dương
    mỗi nhãn, nơi ``gamma_neg = 4`` hạ trọng số mẫu âm quá tay và làm lệch xác
    suất về phía dương — dùng :func:`bce_loss` làm mặc định.

    Ý tưởng: hạ trọng số các mẫu âm **dễ** (model đã đoán đúng là âm với độ tin
    cậy cao) để gradient dồn vào các mẫu dương hiếm và mẫu âm khó.

    Bản cũ cài đặt **ngược hướng**: dùng ``w_neg = (1 - p) ** gamma_neg``, nghĩa
    là mẫu âm dễ (``p ≈ 0``) nhận trọng số ``≈ 1`` (không hề bị hạ) còn mẫu âm
    khó (``p ≈ 1``) nhận trọng số ``≈ 0`` (bị bỏ qua) — đúng ngược lại ý đồ
    (lỗi P0-7). Phần dương cũng sai: ``1 - p ** gamma_pos`` thay vì
    ``(1 - p) ** gamma_pos``; hai biểu thức tình cờ trùng nhau tại
    ``gamma_pos = 1`` nên lỗi bị che giấu ở cấu hình mặc định.

    Args:
        logits: ``(B, C)`` logit thô, chưa qua sigmoid.
        targets: ``(B, C)`` giá trị trong ``{0, 1}``.
        gamma_neg: Số mũ focal cho mẫu âm; càng lớn càng hạ mạnh mẫu âm dễ.
        gamma_pos: Số mũ focal cho mẫu dương.
        clip: Dịch xác suất mẫu âm ("probability shifting") để bỏ hẳn các mẫu âm
            rất dễ ra khỏi loss.
    """
    probs = torch.sigmoid(logits)
    probs_neg = (1.0 - probs + clip).clamp(max=1.0)

    loss_pos = targets * ((1.0 - probs) ** gamma_pos) * torch.log(probs.clamp(min=_EPS))
    loss_neg = (
        (1.0 - targets) * ((1.0 - probs_neg) ** gamma_neg) * torch.log(probs_neg.clamp(min=_EPS))
    )
    return -(loss_pos + loss_neg).sum(dim=1).mean()


def kd_cosine_loss(student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
    """Khoảng cách cosine giữa đặc trưng student và teacher (đã detach).

    Dùng cosine thay vì MSE vì ta muốn khớp *hướng* của biểu diễn chứ không phải
    độ lớn — độ lớn phụ thuộc vào thang đo riêng của từng kiến trúc.
    """
    student = F.normalize(student_feature, dim=-1)
    teacher = F.normalize(teacher_feature.detach(), dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


__all__ = ["asymmetric_loss", "bce_loss", "info_nce", "kd_cosine_loss"]
