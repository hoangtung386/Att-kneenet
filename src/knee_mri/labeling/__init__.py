"""Sinh nhãn yếu 12 chiều từ radiology report.

Hai đường:

* :func:`~knee_mri.labeling.rule_based.keyword_labeler` — luật từ khóa đa ngôn
  ngữ, chạy tức thì trên CPU, không cần model. Dùng làm baseline và fallback.
* :class:`~knee_mri.labeling.llm_based.GemmaReportParser` — VLM đọc hiểu report,
  chính xác hơn nhưng cần GPU. Chỉ dùng ở bước precompute offline.
"""

from knee_mri.labeling.evaluate import (
    LabelerReport,
    cached_weak_labels,
    format_labeler_report,
    score_labeler,
)
from knee_mri.labeling.llm_based import GemmaReportParser, extract_json_labels
from knee_mri.labeling.rule_based import keyword_labeler

__all__ = [
    "GemmaReportParser",
    "LabelerReport",
    "cached_weak_labels",
    "extract_json_labels",
    "format_labeler_report",
    "keyword_labeler",
    "score_labeler",
]
