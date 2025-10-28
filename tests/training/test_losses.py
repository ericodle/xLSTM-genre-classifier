#!/usr/bin/env python3
"""
Test script for loss functions in src/training/losses.py

Tests:
- LabelSmoothingCrossEntropyLoss
- FocalLoss
- Edge cases and numerical stability
- Comparison with standard CrossEntropyLoss
"""

import os
import sys
import unittest
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from training.losses import FocalLoss, LabelSmoothingCrossEntropyLoss


class TestLossFunctions(unittest.TestCase):
    """Test cases for custom loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.num_classes = 10

        # Create mock data
        torch.manual_seed(42)
        self.logits = torch.randn(self.batch_size, self.num_classes, device=self.device)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        # Create perfect predictions for testing
        self.perfect_logits = torch.zeros(self.batch_size, self.num_classes, device=self.device)
        for i in range(self.batch_size):
            self.perfect_logits[i, self.targets[i]] = 10.0  # High confidence

    def test_label_smoothing_cross_entropy_basic(self):
        """Test basic functionality of LabelSmoothingCrossEntropyLoss."""
        smoothing = 0.1
        loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, num_classes=self.num_classes)

        loss = loss_fn(self.logits, self.targets)

        # Basic checks
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)  # Positive loss

        # Test with perfect predictions
        perfect_loss = loss_fn(self.perfect_logits, self.targets)
        self.assertGreater(perfect_loss.item(), 0)  # Should still be positive due to smoothing

    def test_label_smoothing_values(self):
        """Test different smoothing values."""
        # Test smoothing = 0 (should behave like standard cross-entropy)
        loss_fn_no_smooth = LabelSmoothingCrossEntropyLoss(
            smoothing=0.0, num_classes=self.num_classes
        )
        standard_ce = nn.CrossEntropyLoss()

        loss_smooth = loss_fn_no_smooth(self.logits, self.targets)
        loss_standard = standard_ce(self.logits, self.targets)

        # Should be very close (within numerical precision)
        self.assertAlmostEqual(loss_smooth.item(), loss_standard.item(), places=5)

        # Test smoothing = 1.0 (uniform distribution)
        loss_fn_uniform = LabelSmoothingCrossEntropyLoss(
            smoothing=1.0, num_classes=self.num_classes
        )
        loss_uniform = loss_fn_uniform(self.logits, self.targets)

        # Should be positive
        self.assertGreater(loss_uniform.item(), 0)

    def test_label_smoothing_edge_cases(self):
        """Test edge cases for LabelSmoothingCrossEntropyLoss."""
        # Test single class - this should raise an error due to division by zero
        # in the smoothing calculation (smoothing / (num_classes - 1))
        with self.assertRaises(ZeroDivisionError):
            loss_fn_single = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=1)
            single_logits = torch.randn(2, 1, device=self.device)
            single_targets = torch.zeros(2, dtype=torch.long, device=self.device)
            loss_fn_single(single_logits, single_targets)

        # Test two classes
        loss_fn_two = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=2)
        two_logits = torch.randn(2, 2, device=self.device)
        two_targets = torch.tensor([0, 1], device=self.device)

        loss = loss_fn_two(two_logits, two_targets)
        self.assertGreater(loss.item(), 0)

    def test_focal_loss_basic(self):
        """Test basic functionality of FocalLoss."""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

        loss = loss_fn(self.logits, self.targets)

        # Basic checks
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)  # Positive loss

    def test_focal_loss_parameters(self):
        """Test different alpha and gamma values."""
        # Test different alpha values
        for alpha in [0.5, 1.0, 2.0]:
            loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            loss = loss_fn(self.logits, self.targets)
            self.assertGreater(loss.item(), 0)

        # Test different gamma values
        for gamma in [0.5, 1.0, 2.0, 3.0]:
            loss_fn = FocalLoss(alpha=1.0, gamma=gamma)
            loss = loss_fn(self.logits, self.targets)
            self.assertGreater(loss.item(), 0)

    def test_focal_loss_reduction(self):
        """Test different reduction modes."""
        # Test mean reduction (default)
        loss_fn_mean = FocalLoss(reduction="mean")
        loss_mean = loss_fn_mean(self.logits, self.targets)
        self.assertEqual(loss_mean.dim(), 0)

        # Test sum reduction
        loss_fn_sum = FocalLoss(reduction="sum")
        loss_sum = loss_fn_sum(self.logits, self.targets)
        self.assertEqual(loss_sum.dim(), 0)
        self.assertGreater(loss_sum.item(), loss_mean.item())

        # Test none reduction
        loss_fn_none = FocalLoss(reduction="none")
        loss_none = loss_fn_none(self.logits, self.targets)
        self.assertEqual(loss_none.shape, (self.batch_size,))

    def test_focal_loss_vs_cross_entropy(self):
        """Test that FocalLoss behaves differently from CrossEntropyLoss."""
        focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        ce_loss_fn = nn.CrossEntropyLoss()

        focal_loss = focal_loss_fn(self.logits, self.targets)
        ce_loss = ce_loss_fn(self.logits, self.targets)

        # Focal loss should be different from cross-entropy
        self.assertNotAlmostEqual(focal_loss.item(), ce_loss.item(), places=3)

    def test_loss_gradients(self):
        """Test that losses are differentiable."""
        # Test LabelSmoothingCrossEntropyLoss gradients
        logits_grad = self.logits.clone().requires_grad_(True)
        loss_fn_smooth = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=self.num_classes)
        loss_smooth = loss_fn_smooth(logits_grad, self.targets)
        loss_smooth.backward()

        self.assertIsNotNone(logits_grad.grad)
        self.assertFalse(torch.isnan(logits_grad.grad).any())

        # Test FocalLoss gradients
        logits_grad2 = self.logits.clone().requires_grad_(True)
        loss_fn_focal = FocalLoss(alpha=1.0, gamma=2.0)
        loss_focal = loss_fn_focal(logits_grad2, self.targets)
        loss_focal.backward()

        self.assertIsNotNone(logits_grad2.grad)
        self.assertFalse(torch.isnan(logits_grad2.grad).any())

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large logits
        large_logits = torch.randn(self.batch_size, self.num_classes, device=self.device) * 100

        loss_fn_smooth = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=self.num_classes)
        loss_fn_focal = FocalLoss(alpha=1.0, gamma=2.0)

        loss_smooth = loss_fn_smooth(large_logits, self.targets)
        loss_focal = loss_fn_focal(large_logits, self.targets)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(loss_smooth))
        self.assertFalse(torch.isinf(loss_smooth))
        self.assertFalse(torch.isnan(loss_focal))
        self.assertFalse(torch.isinf(loss_focal))

        # Test with very small logits
        small_logits = torch.randn(self.batch_size, self.num_classes, device=self.device) * 0.001

        loss_smooth = loss_fn_smooth(small_logits, self.targets)
        loss_focal = loss_fn_focal(small_logits, self.targets)

        self.assertFalse(torch.isnan(loss_smooth))
        self.assertFalse(torch.isinf(loss_smooth))
        self.assertFalse(torch.isnan(loss_focal))
        self.assertFalse(torch.isinf(loss_focal))

    def test_batch_size_consistency(self):
        """Test that losses work with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            logits = torch.randn(batch_size, self.num_classes, device=self.device)
            targets = torch.randint(0, self.num_classes, (batch_size,), device=self.device)

            loss_fn_smooth = LabelSmoothingCrossEntropyLoss(
                smoothing=0.1, num_classes=self.num_classes
            )
            loss_fn_focal = FocalLoss(alpha=1.0, gamma=2.0)

            loss_smooth = loss_fn_smooth(logits, targets)
            loss_focal = loss_fn_focal(logits, targets)

            self.assertGreater(loss_smooth.item(), 0)
            self.assertGreater(loss_focal.item(), 0)

    def test_class_imbalance_scenario(self):
        """Test losses with imbalanced class distribution."""
        # Create imbalanced targets (mostly class 0)
        imbalanced_targets = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        imbalanced_targets[0] = 1  # Only one sample from class 1

        loss_fn_smooth = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=self.num_classes)
        loss_fn_focal = FocalLoss(alpha=1.0, gamma=2.0)
        loss_fn_ce = nn.CrossEntropyLoss()

        loss_smooth = loss_fn_smooth(self.logits, imbalanced_targets)
        loss_focal = loss_fn_focal(self.logits, imbalanced_targets)
        loss_ce = loss_fn_ce(self.logits, imbalanced_targets)

        # All should be positive
        self.assertGreater(loss_smooth.item(), 0)
        self.assertGreater(loss_focal.item(), 0)
        self.assertGreater(loss_ce.item(), 0)

    def test_loss_monotonicity(self):
        """Test that losses decrease as predictions improve."""
        # Start with random predictions
        logits = torch.randn(4, self.num_classes, device=self.device)
        targets = torch.tensor([0, 1, 2, 3], device=self.device)

        loss_fn_smooth = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=self.num_classes)
        loss_fn_focal = FocalLoss(alpha=1.0, gamma=2.0)

        initial_loss_smooth = loss_fn_smooth(logits, targets)
        initial_loss_focal = loss_fn_focal(logits, targets)

        # Improve predictions by increasing confidence for correct classes
        improved_logits = logits.clone()
        for i in range(4):
            improved_logits[i, targets[i]] += 5.0  # Increase confidence for correct class

        improved_loss_smooth = loss_fn_smooth(improved_logits, targets)
        improved_loss_focal = loss_fn_focal(improved_logits, targets)

        # Improved predictions should have lower loss
        self.assertLess(improved_loss_smooth.item(), initial_loss_smooth.item())
        self.assertLess(improved_loss_focal.item(), initial_loss_focal.item())


def run_loss_tests():
    """Run all loss function tests."""
    print("🧪 Running Loss Function Tests")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLossFunctions)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")

    return success


if __name__ == "__main__":
    success = run_loss_tests()
    sys.exit(0 if success else 1)
