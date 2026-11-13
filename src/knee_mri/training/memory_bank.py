"""Memory bank guidance cho InfoNCE.

InfoNCE cần nhiều mẫu âm, nhưng batch 4–8 study chỉ cho 3–7. Bank giữ lại
guidance của các study đã thấy gần đây để dùng làm mẫu âm thêm.

Bank lưu **vector guidance thô** (đã precompute, cố định) chứ không lưu embedding
đã chiếu. Mỗi bước, trainer chiếu lại cả bank bằng ``text_contrast`` *hiện tại*
(không gradient). Nhờ vậy mẫu âm luôn nhất quán với projector đang học, không có
vấn đề embedding lỗi thời như MoCo phải giải bằng momentum encoder.

Mỗi study chiếm đúng một slot (khóa theo ``study_uid``): gặp lại ở epoch sau thì
ghi đè slot cũ thay vì thêm bản sao, nên bank không chứa bản sao của chính mẫu
dương. Khi tính loss, các slot trùng study với batch hiện tại bị che đi.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class GuidanceBank:
    """Vòng đệm FIFO các vector guidance, khóa theo study."""

    def __init__(self, capacity: int, dim: int, device: str | torch.device = "cpu") -> None:
        if capacity < 1:
            raise ValueError(f"capacity phải >= 1, nhận {capacity}")
        self.capacity = capacity
        self._vectors = torch.zeros(capacity, dim, device=device)
        self._uids: list[str | None] = [None] * capacity
        self._slot_of: dict[str, int] = {}
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._slot_of)

    def push(self, guidance: Tensor, study_uids: Sequence[str]) -> None:
        """Thêm guidance của một batch; study đã có thì ghi đè slot của nó."""
        if guidance.shape[0] != len(study_uids):
            raise ValueError("guidance và study_uids phải cùng độ dài")
        detached = guidance.detach().to(self._vectors.device, self._vectors.dtype)
        for row, uid in zip(detached, study_uids, strict=True):
            slot = self._slot_of.get(uid)
            if slot is None:
                slot = self._cursor
                self._cursor = (self._cursor + 1) % self.capacity
                evicted = self._uids[slot]
                if evicted is not None:
                    del self._slot_of[evicted]
                self._uids[slot] = uid
                self._slot_of[uid] = slot
            self._vectors[slot] = row

    def sample(self, exclude_uids: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Trả về ``(vectors, valid)``.

        ``vectors`` là ``(K, dim)`` gồm mọi slot đã dùng; ``valid`` là ``(B, K)``
        bool, ``False`` khi slot đó là chính study thứ ``b`` của batch.
        """
        filled = [slot for slot, uid in enumerate(self._uids) if uid is not None]
        index = torch.tensor(filled, dtype=torch.long, device=self._vectors.device)
        vectors = self._vectors[index]
        valid = torch.ones(len(exclude_uids), len(filled), dtype=torch.bool, device=vectors.device)
        position = {slot: column for column, slot in enumerate(filled)}
        for row, uid in enumerate(exclude_uids):
            slot = self._slot_of.get(uid)
            if slot is not None:
                valid[row, position[slot]] = False
        return vectors, valid


__all__ = ["GuidanceBank"]
