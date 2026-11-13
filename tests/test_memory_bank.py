"""Kiểm thử :class:`GuidanceBank` và việc nối nó vào vòng huấn luyện."""

from __future__ import annotations

import pytest
import torch

from knee_mri.training.memory_bank import GuidanceBank


def _vec(value: float, dim: int = 4) -> torch.Tensor:
    return torch.full((1, dim), value)


class TestGuidanceBank:
    def test_rejects_bad_capacity(self) -> None:
        with pytest.raises(ValueError):
            GuidanceBank(0, 4)

    def test_fills_then_evicts_oldest(self) -> None:
        bank = GuidanceBank(2, 4)
        for index, uid in enumerate(["a", "b", "c"]):
            bank.push(_vec(float(index)), [uid])
        assert len(bank) == 2
        vectors, valid = bank.sample(["zzz"])
        assert sorted(vectors[:, 0].tolist()) == [1.0, 2.0]  # "a" đã bị đẩy ra
        assert valid.all()

    def test_same_study_overwrites_its_slot(self) -> None:
        """Gặp lại study ở epoch sau không tạo bản sao."""
        bank = GuidanceBank(4, 4)
        bank.push(_vec(1.0), ["a"])
        bank.push(_vec(5.0), ["a"])
        assert len(bank) == 1
        vectors, _ = bank.sample(["x"])
        assert vectors[0, 0].item() == 5.0

    def test_current_study_is_masked(self) -> None:
        bank = GuidanceBank(4, 4)
        bank.push(torch.cat([_vec(1.0), _vec(2.0)]), ["a", "b"])
        vectors, valid = bank.sample(["b", "new"])
        column_b = [i for i, row in enumerate(vectors) if row[0].item() == 2.0][0]
        assert not valid[0, column_b]
        assert valid[0].sum().item() == 1
        assert valid[1].all()

    def test_push_detaches_and_validates_lengths(self) -> None:
        bank = GuidanceBank(2, 4)
        tracked = _vec(1.0).requires_grad_(True)
        bank.push(tracked, ["a"])
        vectors, _ = bank.sample(["x"])
        assert not vectors.requires_grad
        with pytest.raises(ValueError):
            bank.push(torch.zeros(2, 4), ["only-one"])

    def test_empty_bank_samples_nothing(self) -> None:
        vectors, valid = GuidanceBank(3, 4).sample(["a"])
        assert vectors.shape == (0, 4) and valid.shape == (1, 0)
