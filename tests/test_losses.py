"""Test các hàm loss — đặc biệt khóa lại hai lỗi P0-7 và P0-8."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from knee_mri.training.losses import (  # noqa: E402
    asymmetric_loss,
    info_nce,
    kd_cosine_loss,
)


class TestAsymmetricLoss:
    """P0-7: bản cũ cài đặt ASL ngược hướng."""

    def test_easy_negative_costs_less_than_hard_negative(self) -> None:
        """Đây là chính hành vi mà bản cũ làm ngược.

        Mẫu âm dễ (model đoán đúng, p≈0) phải đóng góp loss NHỎ hơn mẫu âm khó
        (model đoán sai, p≈1). Cài đặt cũ dùng ``w_neg = (1-p)**gamma_neg`` cho
        kết quả ngược lại.
        """
        target = torch.zeros(1, 1)
        easy = asymmetric_loss(torch.tensor([[-10.0]]), target)
        hard = asymmetric_loss(torch.tensor([[10.0]]), target)
        assert easy < hard

    def test_easy_positive_costs_less_than_hard_positive(self) -> None:
        target = torch.ones(1, 1)
        easy = asymmetric_loss(torch.tensor([[10.0]]), target)
        hard = asymmetric_loss(torch.tensor([[-10.0]]), target)
        assert easy < hard

    def test_gamma_neg_downweights_easy_negatives(self) -> None:
        """Tăng gamma_neg phải làm mẫu âm dễ đóng góp ÍT hơn nữa.

        Dùng logit -1.0 (p ≈ 0.27, tức trên ngưỡng ``clip``) để phép dịch xác
        suất chưa loại hẳn mẫu — xem
        :meth:`test_probability_shifting_discards_very_easy_negatives`.
        """
        logits = torch.tensor([[-1.0]])
        target = torch.zeros(1, 1)
        weak = asymmetric_loss(logits, target, gamma_neg=0.0)
        strong = asymmetric_loss(logits, target, gamma_neg=4.0)
        assert strong < weak

    def test_probability_shifting_discards_very_easy_negatives(self) -> None:
        """Mẫu âm có ``p < clip`` bị loại hoàn toàn khỏi loss.

        Đây là tác dụng của phép dịch xác suất trong ASL: khi model đã rất chắc
        chắn một mẫu là âm, mẫu đó không còn đóng góp gradient nào nữa. Chính
        tính chất này giúp 12 nhãn cực hiếm (dương tính ~1–2%) không bị chìm
        trong biển mẫu âm.
        """
        target = torch.zeros(1, 1)
        very_easy = asymmetric_loss(torch.tensor([[-6.0]]), target, clip=0.05)
        assert very_easy.item() == pytest.approx(0.0, abs=1e-6)

    def test_gamma_pos_affects_positive_branch(self) -> None:
        """P0-7 phụ: phần dương cũ dùng ``1 - p**g`` thay vì ``(1-p)**g``.

        Hai biểu thức trùng nhau tại gamma_pos=1 nên lỗi bị che giấu; test này
        dùng gamma_pos khác 1 để phát hiện.
        """
        logits = torch.tensor([[2.0]])
        target = torch.ones(1, 1)
        assert asymmetric_loss(logits, target, gamma_pos=0.0) != pytest.approx(
            float(asymmetric_loss(logits, target, gamma_pos=4.0)), rel=1e-3
        )

    def test_is_differentiable(self) -> None:
        logits = torch.randn(4, 12, requires_grad=True)
        asymmetric_loss(logits, torch.zeros(4, 12)).backward()
        assert logits.grad is not None


class TestInfoNce:
    def test_perfect_alignment_is_low(self) -> None:
        embedding = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        aligned = info_nce(embedding, embedding)
        shuffled = info_nce(embedding, embedding.flip(0))
        assert aligned < shuffled

    def test_single_sample_returns_grad_tensor(self) -> None:
        """Batch một mẫu không có cặp âm; phải trả tensor 0 chứ không chia cho 0."""
        embedding = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
        embedding.requires_grad_(True)
        loss = info_nce(embedding, embedding.detach())
        assert torch.is_tensor(loss) and loss.requires_grad
        assert loss.item() == pytest.approx(0.0)


class TestKdCosineLoss:
    def test_identical_features_give_zero(self) -> None:
        feature = torch.randn(4, 32)
        assert kd_cosine_loss(feature, feature).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_features_give_two(self) -> None:
        feature = torch.randn(4, 32)
        assert kd_cosine_loss(feature, -feature).item() == pytest.approx(2.0, abs=1e-5)

    def test_teacher_gradient_is_detached(self) -> None:
        """Teacher đóng băng — gradient không được chảy ngược vào nó."""
        student = torch.randn(4, 32, requires_grad=True)
        teacher = torch.randn(4, 32, requires_grad=True)
        kd_cosine_loss(student, teacher).backward()
        assert student.grad is not None
        assert teacher.grad is None


class TestBceLoss:
    def test_matches_manual_formula(self) -> None:
        from knee_mri.training.losses import bce_loss

        logits = torch.tensor([[0.0, 2.0]])
        targets = torch.tensor([[1.0, 0.0]])
        expected = -torch.log(torch.tensor(0.5)) - torch.log(1 - torch.sigmoid(torch.tensor(2.0)))
        assert torch.isclose(bce_loss(logits, targets), expected)

    def test_confident_wrong_costs_more_than_confident_right(self) -> None:
        from knee_mri.training.losses import bce_loss

        targets = torch.tensor([[1.0]])
        assert bce_loss(torch.tensor([[-4.0]]), targets) > bce_loss(torch.tensor([[4.0]]), targets)

    def test_pos_weight_scales_positive_term(self) -> None:
        from knee_mri.training.losses import bce_loss

        logits, targets = torch.tensor([[0.0]]), torch.tensor([[1.0]])
        weighted = bce_loss(logits, targets, pos_weight=torch.tensor([3.0]))
        assert torch.isclose(weighted, 3 * bce_loss(logits, targets))

    def test_is_differentiable(self) -> None:
        from knee_mri.training.losses import bce_loss

        logits = torch.zeros(2, 3, requires_grad=True)
        bce_loss(logits, torch.ones(2, 3)).backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0


class TestClassificationLossSelection:
    def test_default_is_bce(self, cfg) -> None:
        from knee_mri.training.losses import bce_loss
        from knee_mri.training.trainer import classification_loss

        logits, targets = torch.randn(4, 12), (torch.rand(4, 12) > 0.7).float()
        assert torch.isclose(classification_loss(logits, targets, cfg), bce_loss(logits, targets))

    def test_asl_is_selectable(self, cfg) -> None:
        from dataclasses import replace

        from knee_mri.training.trainer import classification_loss

        asl_cfg = replace(cfg, train=replace(cfg.train, cls_loss="asl"))
        logits, targets = torch.randn(4, 12), (torch.rand(4, 12) > 0.7).float()
        assert not torch.isclose(
            classification_loss(logits, targets, cfg), classification_loss(logits, targets, asl_cfg)
        )

    def test_zero_weighted_branches_get_no_gradient_but_enabled_ones_do(self, cfg) -> None:
        """Mặc định contrast=0: text projector không nhận gradient; bật lên thì có."""
        from dataclasses import replace

        from knee_mri.data.collate import collate_studies
        from knee_mri.training.trainer import build_bundle, compute_losses

        def sample(uid: str) -> dict:
            depth, height, width = cfg.data.target_shape
            return {
                "study_uid": uid,
                "volumes": [torch.randn(depth, height, width).numpy()],
                "teacher_feats": [torch.randn(cfg.model.teacher_dim).numpy()],
                "metadata": [[1.0, 0.0, 0.0, 1.0, 1.0]],
                "guidance": torch.randn(cfg.model.guidance_dim).numpy(),
                "weak_label": (torch.rand(12) > 0.5).float().numpy(),
                "gold_label": None,
            }

        batch = collate_studies([sample("a"), sample("b"), sample("c")])

        def text_grad(config) -> float:
            bundle = build_bundle(config)
            loss, _ = compute_losses(bundle, batch, config, "cpu")
            loss.backward()
            grads = [p.grad for p in bundle.text_contrast.parameters() if p.grad is not None]
            return float(sum(g.abs().sum() for g in grads)) if grads else 0.0

        assert text_grad(cfg) == 0.0
        enabled = replace(cfg, train=replace(cfg.train, loss=replace(cfg.train.loss, contrast=1.0)))
        assert text_grad(enabled) > 0.0


class TestInfoNceWithBank:
    """InfoNCE với mẫu âm bổ sung từ memory bank."""

    @staticmethod
    def _unit(rows: int, dim: int = 16) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.randn(rows, dim), dim=-1)

    def test_empty_bank_equals_plain_infonce(self) -> None:
        image, text = self._unit(4), self._unit(4)
        plain = info_nce(image, text)
        with_empty = info_nce(image, text, negative_text=torch.zeros(0, 16))
        assert with_empty.item() == pytest.approx(plain.item())

    def test_more_negatives_raise_random_loss_floor(self) -> None:
        """Loss ngẫu nhiên là log(B + K): bank làm bài toán khó hơn."""
        torch.manual_seed(0)
        image, text = self._unit(4), self._unit(4)
        plain = info_nce(image, text, temperature=1.0).item()
        banked = info_nce(image, text, temperature=1.0, negative_text=self._unit(64)).item()
        assert banked > plain

    def test_bank_enables_single_sample_batch(self) -> None:
        """Batch 1 study vẫn có tín hiệu nếu bank có mẫu âm."""
        image = self._unit(1).requires_grad_(True)
        loss = info_nce(image, image.detach(), negative_text=self._unit(8))
        loss.backward()
        assert loss.item() > 0 and image.grad is not None

    def test_invalid_entries_are_ignored(self) -> None:
        """Slot bị che (chính study đó) không được tính như mẫu âm."""
        image = self._unit(2)
        bank = image.clone()  # bank chứa đúng các mẫu dương -> sẽ thành âm giả
        valid = ~torch.eye(2, dtype=torch.bool)
        unmasked = info_nce(image, image, negative_text=bank)
        masked = info_nce(image, image, negative_text=bank, negative_valid=valid)
        plain = info_nce(image, image)
        assert unmasked.item() > masked.item() + 0.1
        assert masked.item() == pytest.approx(plain.item(), abs=1e-3)

    def test_bank_receives_no_gradient(self) -> None:
        image = self._unit(3).requires_grad_(True)
        bank = self._unit(5).requires_grad_(True)
        info_nce(image, image.detach(), negative_text=bank).backward()
        assert bank.grad is None
