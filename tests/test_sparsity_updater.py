import unittest

import torch

from sparling import LinearThresholdAdaptiveSUO, NoopSUO


class FakeModel:
    def __init__(self, sparsity_value=0.5):
        self.sparsity_value = sparsity_value


class NoopSUOTest(unittest.TestCase):
    def test_zero_grad_delegates(self):
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        suo = NoopSUO(optimizer)
        # Should not raise
        suo.zero_grad()

    def test_step_delegates(self):
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        suo = NoopSUO(optimizer)
        suo.step()

    def test_update_sparsity_does_nothing(self):
        suo = NoopSUO(optimizer=None)
        model = FakeModel(0.5)
        suo.update_sparsity(model, 0, {})
        self.assertEqual(model.sparsity_value, 0.5)

    def test_update_sparsity_multiple_times(self):
        suo = NoopSUO(optimizer=None)
        model = FakeModel(0.3)
        for step in range(10):
            suo.update_sparsity(model, step, {"acc": 0.9})
        self.assertEqual(model.sparsity_value, 0.3)


class LinearThresholdAdaptiveSUOTest(unittest.TestCase):
    def test_same_threshold(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.8,
            minimal_threshold=0.8,
            maximal_threshold=0.8,
            threshold_decrease_per_iter=1e-5,
            minimal_update_frequency=10,
            information_multiplier=0.1,
            initial_step=0,
        )
        self.assertEqual(model.sparsity_value, 0.5)
        suo.update_sparsity(model, 100, acc_info=dict(acc=0.9))
        self.assertEqual(model.sparsity_value, 0.95)
        suo.update_sparsity(model, 200, acc_info=dict(acc=0.7))
        self.assertEqual(model.sparsity_value, 0.95)
        suo.update_sparsity(model, 300, acc_info=dict(acc=0.9))
        self.assertEqual(model.sparsity_value, 0.995)
        suo.update_sparsity(model, 305, acc_info=dict(acc=0.9))
        self.assertEqual(model.sparsity_value, 0.995)
        suo.update_sparsity(model, 311, acc_info=dict(acc=0.9))
        self.assertEqual(model.sparsity_value, 0.9995)

    def test_updating_threshold(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.9,
            minimal_threshold=0.8,
            maximal_threshold=0.9,
            threshold_decrease_per_iter=0.01,
            minimal_update_frequency=0,
            information_multiplier=0.1,
            initial_step=0,
        )
        self.assertEqual(model.sparsity_value, 0.5)
        suo.update_sparsity(model, 3, acc_info=dict(acc=0.89))  # 89% > 87%
        self.assertEqual(model.sparsity_value, 0.95)
        suo.update_sparsity(
            model, 4, acc_info=dict(acc=0.875)
        )  # threshold updated to 88%, 87.5% is less
        self.assertEqual(model.sparsity_value, 0.95)
        suo.update_sparsity(model, 100, acc_info=dict(acc=0.79))  # threshold should be at 80%
        self.assertEqual(model.sparsity_value, 0.95)
        suo.update_sparsity(model, 101, acc_info=dict(acc=0.83))  # threshold should be at 80%
        self.assertEqual(model.sparsity_value, 0.995)
        suo.update_sparsity(model, 102, acc_info=dict(acc=0.819))  # threshold should be at 82%
        self.assertEqual(model.sparsity_value, 0.995)

    def test_below_threshold_no_update(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.9,
            minimal_threshold=0.0,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.0,
            minimal_update_frequency=0,
            information_multiplier=0.5,
            initial_step=0,
        )
        suo.update_sparsity(model, 1, acc_info=dict(acc=0.8))
        self.assertEqual(model.sparsity_value, 0.5)

    def test_above_threshold_reduces_information(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.5,
            minimal_threshold=0.0,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.0,
            minimal_update_frequency=0,
            information_multiplier=0.5,
            initial_step=0,
        )
        # density = 1 - 0.5 = 0.5, after multiply by 0.5: density = 0.25, sparsity = 0.75
        suo.update_sparsity(model, 1, acc_info=dict(acc=0.6))
        self.assertEqual(model.sparsity_value, 0.75)

    def test_minimal_update_frequency_enforced(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.0,
            minimal_threshold=0.0,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.0,
            minimal_update_frequency=10,
            information_multiplier=0.5,
            initial_step=0,
        )
        # Step 5 is within minimal_update_frequency of initial_step=0
        suo.update_sparsity(model, 5, acc_info=dict(acc=0.9))
        self.assertEqual(model.sparsity_value, 0.5)
        # Step 10 should trigger
        suo.update_sparsity(model, 10, acc_info=dict(acc=0.9))
        self.assertNotEqual(model.sparsity_value, 0.5)

    def test_threshold_decrease_over_time(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.9,
            minimal_threshold=0.0,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.01,
            minimal_update_frequency=0,
            information_multiplier=0.5,
            initial_step=0,
        )
        # At step 50, threshold should decrease by 0.01*50 = 0.5, so threshold = 0.4
        # acc=0.5 > 0.4 should trigger update
        suo.update_sparsity(model, 50, acc_info=dict(acc=0.5))
        self.assertNotEqual(model.sparsity_value, 0.5)

    def test_threshold_clamped_to_minimal(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.9,
            minimal_threshold=0.8,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.1,
            minimal_update_frequency=0,
            information_multiplier=0.5,
            initial_step=0,
        )
        # At step 100, decrease would be 10.0, but clamped to minimal=0.8
        # acc=0.85 > 0.8 should trigger
        suo.update_sparsity(model, 100, acc_info=dict(acc=0.85))
        self.assertNotEqual(model.sparsity_value, 0.5)

    def test_threshold_clamped_to_maximal(self):
        model = FakeModel(0.5)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.5,
            minimal_threshold=0.0,
            maximal_threshold=0.6,
            threshold_decrease_per_iter=-0.1,  # threshold increases
            minimal_update_frequency=0,
            information_multiplier=0.5,
            initial_step=0,
        )
        # At step 100, increase would be 10.0, but clamped to maximal=0.6
        # acc=0.7 > 0.6 should trigger
        suo.update_sparsity(model, 100, acc_info=dict(acc=0.7))
        self.assertNotEqual(model.sparsity_value, 0.5)

    def test_threshold_ratchets_up_to_accuracy(self):
        model = FakeModel(0.0)
        suo = LinearThresholdAdaptiveSUO(
            optimizer=None,
            initial_threshold=0.5,
            minimal_threshold=0.0,
            maximal_threshold=1.0,
            threshold_decrease_per_iter=0.0,
            minimal_update_frequency=0,
            information_multiplier=0.9,
            initial_step=0,
        )
        # After seeing acc=0.8, threshold ratchets up to 0.8
        suo.update_sparsity(model, 1, acc_info=dict(acc=0.8))
        # Now acc=0.7 < 0.8, no update
        model2 = FakeModel(0.5)
        suo.update_sparsity(model2, 2, acc_info=dict(acc=0.7))
        self.assertEqual(model2.sparsity_value, 0.5)
