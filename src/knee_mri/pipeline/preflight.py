"""Kiểm tra môi trường trước khi chạy pipeline nặng.

Mỗi lỗi ở đây đều rẻ để phát hiện lúc đầu nhưng đắt nếu để lộ ra sau hàng giờ
GPU: token HF chưa được cấp quyền model gated, ``guidance_dim`` lệch với VLM thật,
thiếu GPU, hết dung lượng Drive, thiếu weights 3DINO.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from knee_mri.config.schema import Config
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.series_selection import select_series
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

#: Dưới ngưỡng này, Gemma-3-12b bf16 + SAM-huge có nguy cơ OOM.
MIN_GPU_GB: float = 38.0
#: Hệ số dư cho dung lượng mask + file phụ.
DISK_MARGIN: float = 1.3
DISK_FIXED_GB: float = 5.0


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Kết quả kiểm tra: ``problems`` rỗng nghĩa là có thể chạy."""

    problems: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_failed(self) -> None:
        if self.problems:
            raise RuntimeError("Preflight thất bại:\n - " + "\n - ".join(self.problems))


def estimate_mask_gb(cfg: Config, catalog: StudyCatalog, study_uids: Sequence[str]) -> float:
    """Dung lượng mask (GB): uint8 ``(D,H,W)`` không nén cho mỗi series được chọn."""
    n_series = sum(len(select_series(catalog, uid, cfg.data)) for uid in study_uids)
    depth, height, width = cfg.data.target_shape
    return n_series * depth * height * width / 1e9


def check_hf_access(model_ids: Iterable[str], token: str | None) -> list[str]:
    """Thử tải ``config.json`` (vài KB) của từng model để biết token có quyền không."""
    from huggingface_hub import hf_hub_download

    problems: list[str] = []
    for model_id in model_ids:
        try:
            hf_hub_download(model_id, "config.json", token=token)
        except Exception as exc:  # noqa: BLE001 - mọi lỗi mạng/quyền đều đáng báo
            problems.append(
                f"Không truy cập được {model_id} ({type(exc).__name__}). "
                "Mở trang model trên HF, bấm Agree, và kiểm tra HF_TOKEN."
            )
    return problems


def check_guidance_dim(cfg: Config, token: str | None) -> tuple[list[str], int | None]:
    """So ``model.guidance_dim`` với ``hidden_size`` thật của VLM (chỉ tải config)."""
    from transformers import AutoConfig

    try:
        vlm_config = AutoConfig.from_pretrained(cfg.vlm.model_id, token=token)
    except Exception as exc:  # noqa: BLE001
        return [f"Không đọc được config VLM {cfg.vlm.model_id}: {exc}"], None
    hidden = int(getattr(vlm_config, "text_config", vlm_config).hidden_size)
    if hidden != cfg.model.guidance_dim:
        return [
            f"guidance_dim lệch: cấu hình {cfg.model.guidance_dim}, VLM thật {hidden}. "
            f"Ghi đè model.guidance_dim={hidden}."
        ], hidden
    return [], hidden


def run_preflight(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    hf_token: str | None = None,
    require_cuda: bool = True,
    check_hf: bool = True,
) -> PreflightReport:
    """Chạy toàn bộ kiểm tra và gom mọi vấn đề vào một báo cáo.

    Args:
        check_hf: Tắt để bỏ qua các kiểm tra cần mạng (test, môi trường offline).
    """
    import torch

    cfg.paths.ensure()
    problems: list[str] = []
    info: dict[str, Any] = {}

    if require_cuda:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = f"{props.name}, {props.total_memory / 1e9:.0f} GB"
            if props.total_memory < MIN_GPU_GB * 1e9:
                problems.append(f"GPU < {MIN_GPU_GB:.0f}GB: Gemma-3-12b + SAM-huge có thể OOM.")
        else:
            problems.append("Không có GPU — cần runtime GPU (A100).")

    if check_hf:
        problems += check_hf_access((cfg.vlm.model_id, cfg.sam.model_id), hf_token)
        dim_problems, hidden = check_guidance_dim(cfg, hf_token)
        problems += dim_problems
        info["vlm_hidden_size"] = hidden

    if not cfg.dino.weights_path(cfg.paths.artifacts_root).is_file():
        problems.append(f"Thiếu weights 3DINO: {cfg.dino.weights_path(cfg.paths.artifacts_root)}")
    if not cfg.paths.dino_code_dir.is_dir():
        problems.append(f"Thiếu repo 3DINO: {cfg.paths.dino_code_dir}")

    mask_gb = estimate_mask_gb(cfg, catalog, study_uids)
    free_gb = shutil.disk_usage(cfg.paths.artifacts_root).free / 1e9
    needed_gb = mask_gb * DISK_MARGIN + DISK_FIXED_GB
    info.update(mask_gb=mask_gb, free_gb=free_gb, needed_gb=needed_gb)
    if free_gb < needed_gb:
        problems.append(
            f"Dung lượng có thể không đủ: cần ≈ {needed_gb:.0f} GB, còn {free_gb:.0f} GB."
        )

    report = PreflightReport(problems=problems, info=info)
    logger.info("Preflight: %s | %s", "OK" if report.ok else f"{len(problems)} vấn đề", info)
    return report


__all__ = [
    "PreflightReport",
    "check_guidance_dim",
    "check_hf_access",
    "estimate_mask_gb",
    "run_preflight",
]
