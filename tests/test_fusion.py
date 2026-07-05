"""Unit tests for all ACMF fusion strategies.

Tests verify:
  - Output shape and valid probability distribution
  - Strategy-specific behavior (e.g. confidence-guided weights, equal weight equality)
  - Factory function correctness
  - Learnable fusion forward pass and weight output
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fusion.factory import build_fusion_strategy, list_strategies
from fusion.equal_weight import EqualWeightFusion
from fusion.accuracy_weighted import AccuracyWeightedFusion
from fusion.confidence_guided import ConfidenceGuidedFusion
from fusion.learnable_fusion import LearnableFusion


NUM_CLASSES = 5
BATCH = 4


def make_probs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    p1 = F.softmax(torch.randn(BATCH, NUM_CLASSES), dim=-1)
    p2 = F.softmax(torch.randn(BATCH, NUM_CLASSES), dim=-1)
    return p1, p2


def assert_valid_probs(probs: torch.Tensor) -> None:
    assert probs.shape == (BATCH, NUM_CLASSES), f"Bad shape: {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5), \
        f"Probabilities don't sum to 1: {probs.sum(dim=-1)}"
    assert (probs >= 0).all(), "Negative probabilities found"


class TestEqualWeightFusion:
    def test_output_shape_and_validity(self):
        p1, p2 = make_probs()
        fusion = EqualWeightFusion(num_classes=NUM_CLASSES)
        out = fusion(p1, p2)
        assert_valid_probs(out)

    def test_equal_probs_unchanged(self):
        """If both inputs are the same, output should be identical."""
        torch.manual_seed(42)
        p = F.softmax(torch.randn(BATCH, NUM_CLASSES), dim=-1)
        fusion = EqualWeightFusion(num_classes=NUM_CLASSES)
        out = fusion(p, p)
        assert torch.allclose(out, p, atol=1e-6)

    def test_simple_average(self):
        p1, p2 = make_probs()
        fusion = EqualWeightFusion(num_classes=NUM_CLASSES)
        out = fusion(p1, p2)
        expected = (p1 + p2) * 0.5
        assert torch.allclose(out, expected, atol=1e-6)


class TestAccuracyWeightedFusion:
    def test_output_validity(self):
        p1, p2 = make_probs()
        fusion = AccuracyWeightedFusion(num_classes=NUM_CLASSES, acc_224=0.85, acc_384=0.80)
        out = fusion(p1, p2)
        assert_valid_probs(out)

    def test_higher_accuracy_gets_higher_weight(self):
        """Higher-accuracy model's prediction should dominate when predictions differ."""
        p1 = torch.zeros(1, NUM_CLASSES)
        p1[0, 0] = 1.0   # model 1 is 100% confident in class 0
        p2 = torch.zeros(1, NUM_CLASSES)
        p2[0, 1] = 1.0   # model 2 is 100% confident in class 1

        fusion = AccuracyWeightedFusion(num_classes=NUM_CLASSES, acc_224=0.9, acc_384=0.1)
        out = fusion(p1, p2)
        assert out[0, 0] > out[0, 1], "Higher-accuracy model should dominate"

    def test_equal_accuracy_equals_equal_weight(self):
        p1, p2 = make_probs()
        eq = EqualWeightFusion(num_classes=NUM_CLASSES)
        aw = AccuracyWeightedFusion(num_classes=NUM_CLASSES, acc_224=0.5, acc_384=0.5)
        assert torch.allclose(eq(p1, p2), aw(p1, p2), atol=1e-6)

    def test_zero_accuracy_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            AccuracyWeightedFusion(num_classes=NUM_CLASSES, acc_224=0.0, acc_384=0.0)


class TestConfidenceGuidedFusion:
    def test_output_validity(self):
        p1, p2 = make_probs()
        fusion = ConfidenceGuidedFusion(num_classes=NUM_CLASSES)
        out = fusion(p1, p2)
        assert_valid_probs(out)

    def test_confident_model_dominates(self):
        """When one model is certain and the other is uniform, certain model should win."""
        p_certain = torch.zeros(1, NUM_CLASSES)
        p_certain[0, 2] = 1.0   # 100% confident in class 2

        p_uniform = torch.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES)

        fusion = ConfidenceGuidedFusion(num_classes=NUM_CLASSES, eps=1e-8)
        out = fusion(p_certain, p_uniform)
        # Output should be very close to p_certain
        assert out[0, 2] > 0.95, f"Confident model should dominate. Got {out[0, 2]:.4f}"

    def test_get_weights_sum_to_one(self):
        p1, p2 = make_probs()
        fusion = ConfidenceGuidedFusion(num_classes=NUM_CLASSES)
        w1, w2 = fusion.get_weights(p1, p2)
        total = w1 + w2
        assert torch.allclose(total, torch.ones(BATCH), atol=1e-5)


class TestLearnableFusion:
    def test_output_validity(self):
        p1, p2 = make_probs()
        fusion = LearnableFusion(num_classes=NUM_CLASSES, hidden_dim=32)
        out = fusion(p1, p2)
        assert_valid_probs(out)

    def test_initial_weights_near_equal(self):
        """With zero-initialized output layer, weights should be near 0.5/0.5."""
        p1, p2 = make_probs()
        fusion = LearnableFusion(num_classes=NUM_CLASSES, hidden_dim=32)
        w1, w2 = fusion.get_weights(p1, p2)
        assert torch.allclose(w1, torch.full((BATCH,), 0.5), atol=0.01)
        assert torch.allclose(w2, torch.full((BATCH,), 0.5), atol=0.01)

    def test_has_trainable_parameters(self):
        fusion = LearnableFusion(num_classes=NUM_CLASSES)
        params = list(fusion.parameters())
        assert len(params) > 0, "LearnableFusion must have trainable parameters"


class TestFusionFactory:
    def test_all_strategies_build(self):
        for strategy in list_strategies():
            fusion = build_fusion_strategy(strategy, num_classes=NUM_CLASSES)
            p1, p2 = make_probs()
            out = fusion(p1, p2)
            assert_valid_probs(out), f"Strategy {strategy} produced invalid probs"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            build_fusion_strategy("not_a_strategy")

    def test_shape_mismatch_raises(self):
        fusion = EqualWeightFusion(num_classes=NUM_CLASSES)
        p1 = torch.rand(4, NUM_CLASSES)
        p2 = torch.rand(3, NUM_CLASSES)   # different batch size
        with pytest.raises(AssertionError):
            fusion(p1, p2)
