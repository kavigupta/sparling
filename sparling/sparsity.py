from abc import ABC

import numpy as np
import torch
from dconstruct import construct
from torch import nn


class Sparsity(ABC, nn.Module):
    """Abstract base class for all sparsity layers.

    All sparsity layers are ``nn.Module`` subclasses whose target sparsity can be
    updated at any time via the :attr:`sparsity` property.  Subclasses may override
    :meth:`notify_sparsity` to react to changes.

    :param starting_sparsity: initial fraction of values to zero out (0-1).
    :param channels: number of channels the layer operates on.
    """

    def __init__(self, starting_sparsity, channels):
        super().__init__()
        self._sparsity = starting_sparsity
        self.channels = channels

    def notify_sparsity(self):
        """Called whenever :attr:`sparsity` is updated.

        Subclasses should override this to propagate sparsity changes to internal
        components (e.g. underlying enforcers).  The default implementation does nothing.
        """

    @property
    def sparsity(self):
        """The current target sparsity (fraction of values to zero out, 0-1).

        Setting this property updates the internal state and calls :meth:`notify_sparsity`.
        """
        return self._sparsity

    @property
    def density(self):
        """The current density, equal to ``1 - sparsity``."""
        return 1 - self.sparsity

    @sparsity.setter
    def sparsity(self, sparsity):
        self._sparsity = sparsity
        self.notify_sparsity()


class NoSparsity(Sparsity):
    """Identity pass-through — applies no sparsity."""

    def forward(self, x, disable_relu=False):  # pylint: disable=unused-argument
        return x


class SparsityForL1(Sparsity):
    """Applies ReLU activation, intended for use with an external L1 loss to control sparsity."""

    def forward(self, x, disable_relu=False):
        if disable_relu:
            return x
        return torch.nn.functional.relu(x)


class ChangingSparsityForL1(Sparsity):
    """ReLU with density-scaled motif loss.

    Returns a dict with ``result`` (the ReLU output) and ``motifs_for_loss``
    (the output scaled by ``1 / (1 - sparsity)``).
    """

    def forward(self, x, disable_relu=False):
        assert not disable_relu
        x = torch.nn.functional.relu(x)
        mot = x / (1 - self.sparsity)
        return dict(
            result=x,
            motifs_for_loss=mot,
        )


class SparsityForKL(Sparsity):
    """Applies sigmoid activation, intended for use with an external KL loss to control sparsity."""

    def forward(self, x, disable_relu=False):
        assert not disable_relu
        return torch.sigmoid(x)


class EnforceSparsityPerChannel(Sparsity):
    """
    Enforces sparsity across the last index in the given tensor, where C is num_channels.

    Takes an input of size (N, C) and enforces sparsities for each channel C independently.

    :param starting_sparsity: initial sparsity to enforce
    :param channels: the number of channels C to enforce sparsity on
    :param momentum: the momentum of the collected percentile statistic. The default value of 0.1
        indicates that at each batch update we use 90% existing thresholds and 10% the percentile
        statistic thresholds.
    """

    def __init__(
        self,
        starting_sparsity,
        channels,
        momentum=0.1,
    ):
        super().__init__(starting_sparsity, channels)
        self.thresholds = torch.nn.parameter.Parameter(torch.zeros(channels), requires_grad=False)
        self.momentum = momentum

    def update_with_batch(self, x):
        N, _ = x.shape
        to_drop = max(1, int(N * self.sparsity))
        thresholds, _ = torch.kthvalue(x, k=to_drop, dim=0)

        self.thresholds.data = (
            self.thresholds.data * (1 - self.momentum) + thresholds * self.momentum
        )

    def forward(self, x, disable_relu=False):
        _, C = x.shape

        assert [C] == list(self.thresholds.shape), f"{[C]} != {self.thresholds.shape}"
        if self.training:
            self.update_with_batch(x)
        x = x - self.thresholds
        if disable_relu:
            return x
        return torch.nn.functional.relu(x)


class EnforceSparsityPerChannelAccumulated(EnforceSparsityPerChannel):
    """
    Like EnforceSparsityPerChannel, but accumulates the values across batches.
    """

    def __init__(self, *args, accumulation_stop_strategy, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_stop_strategy = construct(
            accumulation_stop_strategy_types(),
            accumulation_stop_strategy,
        )
        self.accumulated_batches = []
        self.num_elements = 0

    def update_with_batch(self, x):
        self.accumulated_batches.append(x.detach())
        self.num_elements += x.shape[0]
        if self.accumulation_stop_strategy.should_stop(self):
            x = torch.cat(self.accumulated_batches, dim=0)
            super().update_with_batch(x)
            self.accumulated_batches = []
            self.num_elements = 0


class StopAtFixedNumberElements:
    """Stop accumulating when the total number of elements reaches a fixed threshold.

    :param num_elements: number of elements to accumulate before triggering a threshold update.
    """

    def __init__(self, num_elements):
        self.num_elements = num_elements

    def should_stop(self, enforcer):
        return enforcer.num_elements >= self.num_elements


class StopAtFixedNumberMotifs:
    """Stop accumulating when the estimated number of elements, multiplied by the current density, reaches a fixed threshold.

    :param num_motifs: target number of motifs (nonzero values) to accumulate before triggering a threshold update.
    """

    def __init__(self, num_motifs):
        self.num_motifs = num_motifs

    def should_stop(self, enforcer):
        return enforcer.num_elements * enforcer.density >= self.num_motifs


def accumulation_stop_strategy_types():
    return dict(
        StopAtFixedNumber=StopAtFixedNumberElements,
        StopAtFixedNumberMotifs=StopAtFixedNumberMotifs,
    )


def enforce_sparsity_per_channel_types():
    return dict(
        EnforceSparsityPerChannel=EnforceSparsityPerChannel,
        EnforceSparsityPerChannelAccumulated=EnforceSparsityPerChannelAccumulated,
    )


class EnforceSparsityPerChannel2D(Sparsity):
    """Like :class:`EnforceSparsityPerChannel`, but for 4-D inputs of shape ``(N, C, H, W)``.

    Reshapes the spatial dimensions into the batch dimension, delegates to an underlying
    :class:`EnforceSparsityPerChannel`, then reshapes back.

    :param starting_sparsity: initial sparsity to enforce.
    :param channels: number of channels C.
    :param momentum: momentum for threshold updates.
    :param enforce_sparsity_per_channel_spec: optional ``dconstruct`` spec for the underlying
        enforcer.  Defaults to ``EnforceSparsityPerChannel``.
    """

    def __init__(
        self,
        starting_sparsity,
        channels,
        momentum=0.1,
        *,
        enforce_sparsity_per_channel_spec=None,
    ):
        super().__init__(starting_sparsity, channels)
        if enforce_sparsity_per_channel_spec is None:
            enforce_sparsity_per_channel_spec = dict(type="EnforceSparsityPerChannel")
        self.channels = channels
        self.underlying_enforcer = construct(
            enforce_sparsity_per_channel_types(),
            enforce_sparsity_per_channel_spec,
            starting_sparsity=starting_sparsity,
            channels=channels,
            momentum=momentum,
        )

    def notify_sparsity(self):
        """Override."""  # noqa: D401
        super().notify_sparsity()
        self.underlying_enforcer.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        N, C, H, W = x.shape
        assert C == self.channels, str((C, self.channels))
        # x : (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        # x : (N, H, W, C)
        x = x.reshape(-1, self.channels)
        # x : (N * H * W, C)
        assert x.shape == (N * H * W, C)
        x = self.underlying_enforcer(x, disable_relu=disable_relu)
        # x : (N * H * W, C)
        x = x.reshape(N, H, W, self.channels)
        # x : (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
        # x : (N, C, H, W)
        assert x.shape == (N, C, H, W)
        return x


class EnforceSparsityPerChannel1D(EnforceSparsityPerChannel2D):
    """Like :class:`EnforceSparsityPerChannel2D`, but for 3-D inputs of shape ``(N, C, L)``."""

    def forward(self, x, **kwargs):
        x = x.unsqueeze(2)
        x = super().forward(x, **kwargs)
        x = x.squeeze(2)
        return x


class EnforceSparsityUniversally(Sparsity):
    """Like :class:`EnforceSparsityPerChannel` but with a single global threshold across all channels.

    :param starting_sparsity: initial sparsity to enforce.
    :param channels: ignored (kept for interface compatibility).
    :param momentum: momentum for threshold updates.
    """

    def __init__(self, starting_sparsity, channels=None, momentum=0.1):
        super().__init__(starting_sparsity, 1)
        self.underlying_enforcer = EnforceSparsityPerChannel(
            starting_sparsity=starting_sparsity, channels=1, momentum=momentum
        )

    def notify_sparsity(self):
        """Override."""  # noqa: D401
        super().notify_sparsity()
        self.underlying_enforcer.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        shape = x.shape
        x = x.reshape(-1, 1)
        x = self.underlying_enforcer(x, disable_relu=disable_relu)
        x = x.reshape(shape)
        return x


class NoiseRatherThanSparsity(Sparsity):
    """Uses noise rather than sparsity to enforce an informational bottleneck.

    Relationship between noise and sparsity::

        information ~= channels * hbern(sparsity)
        information ~= channels / 2 * log(1 + 1 / sigma^2)

        solving for sigma:
            sigma^2 = 1 / (exp(2 * information / channels) - 1)

    This is a very rough approximation:  you should calibrate the actual
    information values empirically.  This just ensures you get roughly the
    right range of values.
    """

    def __init__(self, starting_sparsity, channels=None):
        super().__init__(starting_sparsity, 1)
        # will be applied across channels anyway
        self.norm = nn.BatchNorm1d(1, affine=False)

    @property
    def sigma(self):
        s = self.sparsity
        information_over_channels = -(s * np.log(s) + (1 - s) * np.log(1 - s))
        sigma = np.sqrt(1 / (np.exp(2 * information_over_channels) - 1))
        return sigma

    def forward(self, x, disable_relu=False):
        assert not disable_relu
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, 1, -1)
        x = self.norm(x)
        x = x.reshape(batch_size, channels, height, width)
        x = x + torch.randn_like(x) * self.sigma
        return x


class SparseLayerWithBatchNorm(Sparsity):
    """Wraps a sparsity enforcer with a preceding batch normalization layer.

    This is the recommended way to use Sparling — the batch norm is necessary for the
    sparsity enforcement algorithm to work well in practice.

    :param underlying_sparsity_spec: ``dconstruct`` spec dict for the inner sparsity layer
        (e.g. ``dict(type="EnforceSparsityPerChannel2D")``).
    :param starting_sparsity: initial sparsity to enforce.
    :param channels: number of channels.
    :param affine: whether the batch norm has learnable affine parameters.
    :param input_dimensions: ``2`` for ``(N, C, H, W)`` inputs, ``1`` for ``(N, C, L)``.
    """

    def __init__(
        self,
        underlying_sparsity_spec,
        *,
        starting_sparsity,
        channels,
        affine,
        input_dimensions=2,
    ):
        super().__init__(starting_sparsity, channels)
        self.input_dimensions = input_dimensions
        self.batch_norm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}[input_dimensions](
            channels, affine=affine
        )
        self.underlying_enforcer = construct(
            sparsity_types(),
            underlying_sparsity_spec,
            starting_sparsity=starting_sparsity,
            channels=channels,
        )

    def notify_sparsity(self):
        """Override."""  # noqa: D401
        super().notify_sparsity()
        self.underlying_enforcer.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        if getattr(self, "input_dimensions", 2) == 1:
            x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = self.underlying_enforcer(x, disable_relu=disable_relu)
        if getattr(self, "input_dimensions", 2) == 1:
            x = x.transpose(1, 2)
        return x


class ParallelSparsityLayers(Sparsity):
    """Applies different sparsity layers to disjoint channel subsets in parallel.

    :param sparse_model_specs: list of ``dconstruct`` spec dicts, one per subset.
    :param channels_each: list of channel counts for each subset (must sum to *channels*).
    :param starting_sparsity: initial sparsity to enforce.
    :param channels: total number of channels.
    """

    def __init__(self, sparse_model_specs, channels_each, starting_sparsity, channels):
        super().__init__(starting_sparsity=starting_sparsity, channels=channels)
        assert len(sparse_model_specs) == len(channels_each)
        assert sum(channels_each) == channels, (channels_each, channels)
        self.channels_each = channels_each
        self.sparse_layers = nn.ModuleList(
            [
                construct(
                    sparsity_types(),
                    sparsity_spec,
                    starting_sparsity=starting_sparsity,
                    channels=channels_this,
                )
                for sparsity_spec, channels_this in zip(sparse_model_specs, channels_each)
            ]
        )

    def notify_sparsity(self):
        """Override."""  # noqa: D401
        super().notify_sparsity()
        for starting_sparsity in self.sparse_layers:
            starting_sparsity.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        out = []
        start_idx = 0
        assert x.size(2) == sum(self.channels_each), (x.size(2), self.channels_each)
        for sparse_layer, num_channels in zip(self.sparse_layers, self.channels_each):
            x_selected = x[:, :, start_idx : start_idx + num_channels]
            start_idx += num_channels
            out.append(sparse_layer(x_selected, disable_relu=disable_relu))
        out = torch.cat(out, dim=2)
        return out


def sparsity_types():
    return dict(
        NoSparsity=NoSparsity,
        SparsityForL1=SparsityForL1,
        ChangingSparsityForL1=ChangingSparsityForL1,
        SparsityForKL=SparsityForKL,
        EnforceSparsityPerChannel1D=EnforceSparsityPerChannel1D,
        EnforceSparsityPerChannel2D=EnforceSparsityPerChannel2D,
        EnforceSparsityUniversally=EnforceSparsityUniversally,
        NoiseRatherThanSparsity=NoiseRatherThanSparsity,
        SparseLayerWithBatchNorm=SparseLayerWithBatchNorm,
        ParallelSparsityLayers=ParallelSparsityLayers,
    )
