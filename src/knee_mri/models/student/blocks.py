"""Khối transformer trả về ma trận attention để tính CAM.

Bản cũ dùng ``nn.TransformerEncoderLayer`` rồi cố móc attention bằng forward
hook trên ``layer.self_attn``. Cách đó **không bao giờ hoạt động**: lớp đó gọi
``self_attn(..., need_weights=False)``, nên phần tử thứ hai của tuple trả về luôn
là ``None`` và danh sách hook luôn rỗng (lỗi P0-3). Hệ quả dây chuyền là toàn bộ
tính năng CAM ở S3 trở thành code chết, và biến ``cam`` chưa gán gây
``UnboundLocalError`` (lỗi P0-4).

Ở đây khối tự gọi attention nên luôn kiểm soát được ``need_weights``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    """Khối transformer pre-norm, tùy chọn trả về trọng số attention.

    Dùng pre-norm (LayerNorm trước attention/MLP) vì nó ổn định hơn hẳn khi
    huấn luyện ViT sâu mà không cần lịch warmup phức tạp.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, *, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        """Chạy khối.

        Args:
            x: Tensor ``(B, N, D)``.
            return_attn: Có trả về trọng số attention hay không. Đặt ``False``
                lúc huấn luyện để tiết kiệm bộ nhớ (ma trận ``N × N`` cho mỗi
                head là rất tốn).

        Returns:
            ``(output, attn)`` với ``attn`` có shape ``(B, heads, N, N)`` khi
            ``return_attn=True``, ngược lại là ``None``.
        """
        normed = self.norm_attn(x)
        attended, weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.norm_mlp(x))
        return x, weights


def sinusoidal_position_embedding(num_positions: int, dim: int) -> Tensor:
    """Positional embedding sin/cos cố định, dùng để khởi tạo ``pos_embed``.

    Khởi tạo có cấu trúc hội tụ nhanh hơn khởi tạo ngẫu nhiên thuần, trong khi
    tham số vẫn học được tiếp.
    """
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    embedding = torch.zeros(num_positions, dim)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term[: embedding[:, 1::2].shape[1]])
    return embedding.unsqueeze(0)


def sinusoidal_position_embedding_3d(grid: tuple[int, int, int], dim: int) -> Tensor:
    """Positional embedding sin/cos **tách theo ba trục** của lưới patch.

    Đánh số phẳng 1D khiến hai patch kề nhau theo trục W cách nhau 1 vị trí còn
    hai patch kề nhau theo trục D cách nhau ``H·W`` vị trí — model phải tự học
    lại hình học 3D từ dữ liệu vốn đã ít. Ở đây mỗi trục có một khối ``dim//3``
    chiều riêng, ghép lại và đệm 0 cho phần dư. Hàng đầu là vị trí của token
    CLS (toàn 0).

    Returns:
        Tensor ``(1, 1 + D·H·W, dim)``, thứ tự patch khớp ``Conv3d.flatten(2)``.
    """
    axis_dim = (dim // 3) // 2 * 2
    if axis_dim < 2:
        raise ValueError(f"dim={dim} quá nhỏ để chia cho ba trục.")
    depth, height, width = grid
    along_d = sinusoidal_position_embedding(depth, axis_dim)[0][:, None, None, :]
    along_h = sinusoidal_position_embedding(height, axis_dim)[0][None, :, None, :]
    along_w = sinusoidal_position_embedding(width, axis_dim)[0][None, None, :, :]
    patches = torch.cat(
        [
            along_d.expand(depth, height, width, -1),
            along_h.expand(depth, height, width, -1),
            along_w.expand(depth, height, width, -1),
        ],
        dim=-1,
    )
    patches = F.pad(patches, (0, dim - 3 * axis_dim)).reshape(depth * height * width, dim)
    return torch.cat([torch.zeros(1, dim), patches], dim=0).unsqueeze(0)


__all__ = [
    "AttentionBlock",
    "sinusoidal_position_embedding",
    "sinusoidal_position_embedding_3d",
]
