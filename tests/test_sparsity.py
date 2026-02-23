import unittest

import torch

from sparling.sparsity import (
    ChangingSparsityForL1,
    EnforceSparsity1D,
    EnforceSparsityPerChannel,
    EnforceSparsityPerChannel1D,
    EnforceSparsityPerChannel2D,
    EnforceSparsityPerChannelAccumulated,
    EnforceSparsityUniversally,
    NoiseRatherThanSparsity,
    NoSparsity,
    ParallelSparsityLayers,
    SparseLayerWithBatchNorm,
    SparsityForKL,
    SparsityForL1,
)

CHANNELS = 32
NUM_BATCHES = 100
TOLERANCE = 0.03


def _empirical_sparsity(output):
    return (output == 0).float().mean().item()


class NoSparsityTest(unittest.TestCase):
    def test_forward_identity(self):
        s = NoSparsity(starting_sparsity=0.5, channels=CHANNELS)
        x = torch.randn(1000, CHANNELS)
        self.assertTrue(torch.equal(s(x), x))
        self.assertTrue(torch.equal(s(x, disable_relu=True), x))


class SparsityForL1Test(unittest.TestCase):
    def test_forward_applies_relu(self):
        s = SparsityForL1(starting_sparsity=0.5, channels=CHANNELS)
        x = torch.randn(1000, CHANNELS)
        out = s(x)
        self.assertTrue(torch.equal(out, torch.relu(x)))

    def test_forward_disable_relu(self):
        s = SparsityForL1(starting_sparsity=0.5, channels=CHANNELS)
        x = torch.randn(1000, CHANNELS)
        self.assertTrue(torch.equal(s(x, disable_relu=True), x))


class ChangingSparsityForL1Test(unittest.TestCase):
    def test_result_and_motifs(self):
        s = ChangingSparsityForL1(starting_sparsity=0.3, channels=CHANNELS)
        x = torch.randn(1000, CHANNELS)
        out = s(x)
        self.assertTrue(torch.equal(out["result"], torch.relu(x)))
        # motifs_for_loss = relu(x) / density = relu(x) / 0.7
        self.assertTrue(torch.allclose(out["motifs_for_loss"], torch.relu(x) / 0.7))


class SparsityForKLTest(unittest.TestCase):
    def test_forward_applies_sigmoid(self):
        s = SparsityForKL(starting_sparsity=0.5, channels=CHANNELS)
        x = torch.randn(1000, CHANNELS)
        out = s(x)
        self.assertTrue(torch.allclose(out, torch.sigmoid(x)))


class EnforceSparsityPerChannelTest(unittest.TestCase):
    def _calibrate(self, sparsity, channels=CHANNELS, batch_size=500, num_batches=NUM_BATCHES):
        torch.manual_seed(0)
        s = EnforceSparsityPerChannel(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(num_batches):
            s(torch.randn(batch_size, channels))
        s.eval()
        return s

    def _calibrate_heterogeneous(self, sparsity, channels=CHANNELS, batch_size=500):
        """Calibrate with different mean per channel: means range from -2 to 2."""
        torch.manual_seed(0)
        means = torch.linspace(-2, 2, channels)
        s = EnforceSparsityPerChannel(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(batch_size, channels) + means)
        s.eval()
        return s, means

    def _check_per_channel(self, out, target):
        sparsities = (out == 0).float().mean(dim=0)
        for ch, sp in enumerate(sparsities):
            self.assertAlmostEqual(
                sp.item(), target, delta=TOLERANCE, msg=f"channel {ch}: {sp.item():.3f}"
            )

    def test_sparsity_30(self):
        s = self._calibrate(0.3)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self._check_per_channel(out, 0.3)

    def test_sparsity_70(self):
        s = self._calibrate(0.7)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self._check_per_channel(out, 0.7)

    def test_sparsity_90(self):
        s = self._calibrate(0.9)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self._check_per_channel(out, 0.9)

    def test_sparsity_95(self):
        s = self._calibrate(0.95)
        torch.manual_seed(99)
        out = s(torch.randn(10000, CHANNELS))
        self._check_per_channel(out, 0.95)

    def test_heterogeneous_per_channel_sparsity(self):
        """Channels with different distributions should still hit target sparsity per channel."""
        s, means = self._calibrate_heterogeneous(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS) + means)
        self._check_per_channel(out, 0.5)

    def test_thresholds_approximate_quantile(self):
        """For N(0,1) input, threshold should approximate the normal quantile."""
        for target_sparsity in [0.3, 0.5, 0.7, 0.9]:
            s = self._calibrate(target_sparsity)
            expected = torch.distributions.Normal(0, 1).icdf(torch.tensor(target_sparsity)).item()
            for ch, t in enumerate(s.thresholds.data):
                self.assertAlmostEqual(
                    t.item(),
                    expected,
                    delta=0.1,
                    msg=f"sparsity={target_sparsity}, channel {ch}",
                )

    def test_heterogeneous_thresholds_approximate_quantile(self):
        """For N(mu, 1) at 50% sparsity, threshold should approximate mu per channel."""
        s, means = self._calibrate_heterogeneous(0.5)
        for ch, (t, mu) in enumerate(zip(s.thresholds.data, means)):
            self.assertAlmostEqual(t.item(), mu.item(), delta=0.1, msg=f"channel {ch}")

    def test_disable_relu_preserves_negatives(self):
        s = self._calibrate(0.5)
        torch.manual_seed(99)
        x = torch.randn(5000, CHANNELS)
        out = s(x, disable_relu=True)
        self.assertTrue((out < 0).any())
        out_relu = s(x)
        self.assertGreater(_empirical_sparsity(out_relu), _empirical_sparsity(out) + 0.2)

    def test_thresholds_frozen_in_eval(self):
        s = self._calibrate(0.5)
        thresholds_before = s.thresholds.data.clone()
        s(torch.randn(5000, CHANNELS) * 100)
        self.assertTrue(torch.equal(s.thresholds.data, thresholds_before))

    def test_higher_sparsity_zeros_more(self):
        s_low = self._calibrate(0.3)
        s_high = self._calibrate(0.9)
        torch.manual_seed(99)
        x = torch.randn(5000, CHANNELS)
        self.assertGreater(_empirical_sparsity(s_high(x)), _empirical_sparsity(s_low(x)))


class EnforceSparsityPerChannelAccumulatedTest(unittest.TestCase):
    def _calibrate(self, sparsity, num_elements_trigger=2000, num_batches=NUM_BATCHES):
        torch.manual_seed(0)
        s = EnforceSparsityPerChannelAccumulated(
            starting_sparsity=sparsity,
            channels=CHANNELS,
            momentum=0.1,
            accumulation_stop_strategy=dict(
                type="StopAtFixedNumber", num_elements=num_elements_trigger
            ),
        )
        s.train()
        for _ in range(num_batches):
            s(torch.randn(500, CHANNELS))
        s.eval()
        return s

    def _calibrate_heterogeneous(self, sparsity, num_elements_trigger=500):
        torch.manual_seed(0)
        means = torch.linspace(-2, 2, CHANNELS)
        s = EnforceSparsityPerChannelAccumulated(
            starting_sparsity=sparsity,
            channels=CHANNELS,
            momentum=0.1,
            accumulation_stop_strategy=dict(
                type="StopAtFixedNumber", num_elements=num_elements_trigger
            ),
        )
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(500, CHANNELS) + means)
        s.eval()
        return s, means

    def _check_per_channel(self, out, target):
        sparsities = (out == 0).float().mean(dim=0)
        for ch, sp in enumerate(sparsities):
            self.assertAlmostEqual(
                sp.item(), target, delta=TOLERANCE, msg=f"channel {ch}: {sp.item():.3f}"
            )

    def test_sparsity_50(self):
        s = self._calibrate(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self._check_per_channel(out, 0.5)

    def test_sparsity_80(self):
        s = self._calibrate(0.8)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self._check_per_channel(out, 0.8)

    def test_sparsity_95(self):
        s = self._calibrate(0.95)
        torch.manual_seed(99)
        out = s(torch.randn(10000, CHANNELS))
        self._check_per_channel(out, 0.95)

    # def test_sparsity_999(self):
    #     s = self._calibrate(0.999, num_batches=NUM_BATCHES * 100)
    #     expected_thresh = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.999)).item()
    #     for thresh in s.thresholds.data:
    #         self.assertAlmostEqual(thresh.item(), expected_thresh, delta=0.1)

    def test_heterogeneous_per_channel_sparsity(self):
        s, means = self._calibrate_heterogeneous(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(10000, CHANNELS) + means)
        self._check_per_channel(out, 0.5)

    def test_heterogeneous_thresholds_approximate_quantile(self):
        s, means = self._calibrate_heterogeneous(0.5)
        for ch, (t, mu) in enumerate(zip(s.thresholds.data, means)):
            self.assertAlmostEqual(t.item(), mu.item(), delta=0.15, msg=f"channel {ch}")

    def test_accumulation_delays_update(self):
        s = EnforceSparsityPerChannelAccumulated(
            starting_sparsity=0.5,
            channels=CHANNELS,
            momentum=1.0,
            accumulation_stop_strategy=dict(type="StopAtFixedNumber", num_elements=2000),
        )
        s.train()
        # 500 elements, well below 2000 trigger
        s(torch.randn(500, CHANNELS))
        self.assertTrue(torch.equal(s.thresholds.data, torch.zeros(CHANNELS)))
        self.assertEqual(s.num_elements, 500)


class EnforceSparsityPerChannel2DTest(unittest.TestCase):
    def _calibrate(self, sparsity, channels=CHANNELS):
        torch.manual_seed(0)
        s = EnforceSparsityPerChannel2D(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, channels, 8, 8))
        s.eval()
        return s

    def _calibrate_heterogeneous(self, sparsity, channels=CHANNELS):
        torch.manual_seed(0)
        means = torch.linspace(-2, 2, channels).view(1, channels, 1, 1)
        s = EnforceSparsityPerChannel2D(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, channels, 8, 8) + means)
        s.eval()
        return s, means.view(channels)

    def _check_per_channel(self, out, target):
        # out: (N, C, H, W)
        sparsities = (out == 0).float().mean(dim=(0, 2, 3))
        for ch, sp in enumerate(sparsities):
            self.assertAlmostEqual(
                sp.item(), target, delta=TOLERANCE, msg=f"channel {ch}: {sp.item():.3f}"
            )

    def test_sparsity_50(self):
        s = self._calibrate(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 8, 8))
        self._check_per_channel(out, 0.5)

    def test_sparsity_80(self):
        s = self._calibrate(0.8)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 8, 8))
        self._check_per_channel(out, 0.8)

    def test_heterogeneous_per_channel_sparsity(self):
        s, means = self._calibrate_heterogeneous(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 8, 8) + means.view(1, CHANNELS, 1, 1))
        self._check_per_channel(out, 0.5)

    def test_heterogeneous_thresholds_approximate_quantile(self):
        s, means = self._calibrate_heterogeneous(0.5)
        thresholds = s.underlying_enforcer.thresholds.data
        for ch, (t, mu) in enumerate(zip(thresholds, means)):
            self.assertAlmostEqual(t.item(), mu.item(), delta=0.1, msg=f"channel {ch}")

    def test_sparsity_propagates(self):
        s = EnforceSparsityPerChannel2D(starting_sparsity=0.5, channels=CHANNELS)
        s.sparsity = 0.9
        self.assertEqual(s.underlying_enforcer.sparsity, 0.9)


class EnforceSparsityPerChannel1DTest(unittest.TestCase):
    def _calibrate(self, sparsity, channels=CHANNELS):
        torch.manual_seed(0)
        s = EnforceSparsityPerChannel1D(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, channels, 64))
        s.eval()
        return s

    def _calibrate_heterogeneous(self, sparsity, channels=CHANNELS):
        torch.manual_seed(0)
        means = torch.linspace(-2, 2, channels).view(1, channels, 1)
        s = EnforceSparsityPerChannel1D(starting_sparsity=sparsity, channels=channels, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, channels, 64) + means)
        s.eval()
        return s, means.view(channels)

    def _check_per_channel(self, out, target):
        # out: (N, C, L)
        sparsities = (out == 0).float().mean(dim=(0, 2))
        for ch, sp in enumerate(sparsities):
            self.assertAlmostEqual(
                sp.item(), target, delta=TOLERANCE, msg=f"channel {ch}: {sp.item():.3f}"
            )

    def test_sparsity_50(self):
        s = self._calibrate(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 64))
        self._check_per_channel(out, 0.5)

    def test_sparsity_80(self):
        s = self._calibrate(0.8)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 64))
        self._check_per_channel(out, 0.8)

    def test_heterogeneous_per_channel_sparsity(self):
        s, means = self._calibrate_heterogeneous(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(256, CHANNELS, 64) + means.view(1, CHANNELS, 1))
        self._check_per_channel(out, 0.5)


class EnforceSparsityUniversallyTest(unittest.TestCase):
    def _calibrate(self, sparsity):
        torch.manual_seed(0)
        s = EnforceSparsityUniversally(starting_sparsity=sparsity, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(500, CHANNELS))
        s.eval()
        return s

    def test_sparsity_50(self):
        s = self._calibrate(0.5)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self.assertAlmostEqual(_empirical_sparsity(out), 0.5, delta=TOLERANCE)

    def test_sparsity_80(self):
        s = self._calibrate(0.8)
        torch.manual_seed(99)
        out = s(torch.randn(5000, CHANNELS))
        self.assertAlmostEqual(_empirical_sparsity(out), 0.8, delta=TOLERANCE)

    def test_works_on_4d(self):
        torch.manual_seed(0)
        s = EnforceSparsityUniversally(starting_sparsity=0.7, momentum=0.1)
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, 8, 4, 4))
        s.eval()
        torch.manual_seed(99)
        out = s(torch.randn(64, 8, 4, 4))
        self.assertAlmostEqual(_empirical_sparsity(out), 0.7, delta=TOLERANCE)

    def test_threshold_approximate_quantile(self):
        s = self._calibrate(0.7)
        expected = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.7)).item()
        actual = s.underlying_enforcer.thresholds.data[0].item()
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_sparsity_propagates(self):
        s = EnforceSparsityUniversally(starting_sparsity=0.5)
        s.sparsity = 0.8
        self.assertEqual(s.underlying_enforcer.sparsity, 0.8)


class NoiseRatherThanSparsityTest(unittest.TestCase):
    def test_output_deterministic_with_seed(self):
        s = NoiseRatherThanSparsity(starting_sparsity=0.5)
        s.train()
        x = torch.randn(16, 8, 8, 8)
        torch.manual_seed(42)
        out1 = s(x)
        torch.manual_seed(42)
        out2 = s(x)
        self.assertTrue(torch.equal(out1, out2))

    def test_noise_changes_with_seed(self):
        s = NoiseRatherThanSparsity(starting_sparsity=0.5)
        s.train()
        x = torch.randn(16, 8, 8, 8)
        torch.manual_seed(0)
        out1 = s(x)
        torch.manual_seed(1)
        out2 = s(x)
        self.assertFalse(torch.equal(out1, out2))

    def test_sigma_minimal_at_half(self):
        s = NoiseRatherThanSparsity(starting_sparsity=0.5)
        sigma_half = s.sigma
        s.sparsity = 0.1
        self.assertGreater(s.sigma, sigma_half)
        s.sparsity = 0.9
        self.assertGreater(s.sigma, sigma_half)


class SparseLayerWithBatchNormTest(unittest.TestCase):
    def test_calibrated_2d_sparsity(self):
        torch.manual_seed(0)
        s = SparseLayerWithBatchNorm(
            underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
            starting_sparsity=0.7,
            channels=CHANNELS,
            affine=True,
            input_dimensions=2,
        )
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, CHANNELS, 8, 8))
        s.eval()
        torch.manual_seed(99)
        out = s(torch.randn(64, CHANNELS, 8, 8))
        self.assertAlmostEqual(_empirical_sparsity(out), 0.7, delta=TOLERANCE)

    def test_sparsity_propagates(self):
        s = SparseLayerWithBatchNorm(
            underlying_sparsity_spec=dict(type="NoSparsity"),
            starting_sparsity=0.5,
            channels=CHANNELS,
            affine=True,
        )
        s.sparsity = 0.9
        self.assertEqual(s.underlying_enforcer.sparsity, 0.9)


class EnforceSparsity1DTest(unittest.TestCase):
    def test_calibrated_sparsity(self):
        torch.manual_seed(0)
        s = EnforceSparsity1D(
            underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
            starting_sparsity=0.7,
            channels=CHANNELS,
        )
        s.train()
        for _ in range(NUM_BATCHES):
            s(torch.randn(16, CHANNELS, 64))
        s.eval()
        torch.manual_seed(99)
        out = s(torch.randn(64, CHANNELS, 64))
        self.assertAlmostEqual(_empirical_sparsity(out), 0.7, delta=TOLERANCE)

    def test_sparsity_propagates(self):
        s = EnforceSparsity1D(
            underlying_sparsity_spec=dict(type="NoSparsity"),
            starting_sparsity=0.5,
            channels=CHANNELS,
        )
        s.sparsity = 0.8
        self.assertEqual(s.underlying_enforcer.sparsity, 0.8)


class ParallelSparsityLayersTest(unittest.TestCase):
    def test_different_layers_different_behavior(self):
        s = ParallelSparsityLayers(
            sparse_model_specs=[dict(type="NoSparsity"), dict(type="SparsityForL1")],
            channels_each=[16, 16],
            starting_sparsity=0.5,
            channels=32,
        )
        x = torch.randn(100, 10, 32) - 1.0  # shift negative so most values are < 0
        out = s(x)
        # First 16: NoSparsity preserves negatives
        self.assertTrue((out[:, :, :16] < 0).any())
        # Second 16: SparsityForL1 applies ReLU, all >= 0
        self.assertTrue((out[:, :, 16:] >= 0).all())
        # ReLU half should have many zeros since input is shifted negative
        self.assertGreater(_empirical_sparsity(out[:, :, 16:]), 0.5)

    def test_sparsity_propagates(self):
        s = ParallelSparsityLayers(
            sparse_model_specs=[dict(type="NoSparsity"), dict(type="NoSparsity")],
            channels_each=[16, 16],
            starting_sparsity=0.5,
            channels=32,
        )
        s.sparsity = 0.9
        for layer in s.sparse_layers:
            self.assertEqual(layer.sparsity, 0.9)
