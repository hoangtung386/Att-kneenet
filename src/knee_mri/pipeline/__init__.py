"""Các bước của pipeline, mỗi bước là một hàm gọi được độc lập.

Thay cho ``src/pipeline.py`` cũ vốn trộn lẫn orchestration, logic nghiệp vụ và
phân tích tham số dòng lệnh trong cùng một file.

Thứ tự chạy::

    generate_masks              (S1)
    precompute_weak_labels      (S1)
    precompute_teacher_features (S2, chuẩn bị)
    precompute_guidance         (S2, chuẩn bị)
    train                       (S2)
    explain_studies             (S3)
    predict_submission          (submit)
"""

from knee_mri.pipeline.explain_step import (
    CamResult,
    ExplanationResult,
    compute_cam,
    explain_studies,
)
from knee_mri.pipeline.features import precompute_guidance, precompute_teacher_features
from knee_mri.pipeline.masks import generate_masks
from knee_mri.pipeline.preflight import PreflightReport, estimate_mask_gb, run_preflight
from knee_mri.pipeline.runner import StageTimer, estimate_full_run
from knee_mri.pipeline.weak_labels import precompute_weak_labels

__all__ = [
    "CamResult",
    "ExplanationResult",
    "PreflightReport",
    "StageTimer",
    "compute_cam",
    "estimate_full_run",
    "estimate_mask_gb",
    "explain_studies",
    "generate_masks",
    "precompute_guidance",
    "precompute_teacher_features",
    "precompute_weak_labels",
    "run_preflight",
]
