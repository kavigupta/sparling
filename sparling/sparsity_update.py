from abc import ABC, abstractmethod


class SparsityUpdateOptimizer(ABC):
    """Abstract base class for sparsity update optimizers.

    Wraps a standard ``torch`` optimizer and adds a :meth:`update_sparsity` hook
    that adjusts a model's sparsity during training.  Not a ``torch`` optimizer
    itself, but exposes the same ``zero_grad``/``step`` interface.

    :param optimizer: the underlying ``torch`` optimizer to wrap.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def step(self):
        return self._optimizer.step()

    @abstractmethod
    def update_sparsity(self, model, step, acc_info):
        """
        Updates the model's sparsity given the current step and accuracy information
        """


class NoopSUO(SparsityUpdateOptimizer):
    """A no-op sparsity update optimizer.  Useful as a baseline that never changes sparsity."""

    def update_sparsity(self, model, step, acc_info):
        pass


class LinearThresholdAdaptiveSUO(SparsityUpdateOptimizer):
    def __init__(
        self,
        optimizer,
        *,
        initial_threshold,
        minimal_threshold,
        maximal_threshold,
        threshold_decrease_per_iter,
        minimal_update_frequency,
        information_multiplier,
        initial_step=0,
    ):
        """Accuracy-threshold-driven adaptive sparsity update optimizer.

        Maintains a current accuracy threshold that slowly decreases over time.  When the
        model exceeds the threshold, the threshold is raised to match and the model's
        information (``1 - sparsity``) is reduced by ``information_multiplier``.

        :param optimizer: the underlying ``torch`` optimizer.
        :param initial_threshold: the initial accuracy threshold (0-1).
        :param minimal_threshold: the lower bound for the accuracy threshold (0-1).
        :param maximal_threshold: the upper bound for the accuracy threshold (0-1).
        :param threshold_decrease_per_iter: amount the threshold decreases per iteration.
        :param minimal_update_frequency: minimum number of iterations between sparsity updates.
        :param information_multiplier: multiplicative factor applied to information
            (``1 - sparsity``) when the model exceeds the threshold.
        :param initial_step: starting step counter (default 0).
        """
        super().__init__(optimizer)
        self._threshold = initial_threshold
        self._minimal_threshold = minimal_threshold
        self._maximal_threshold = maximal_threshold
        self._threshold_decrease_per_iter = threshold_decrease_per_iter
        self._step = initial_step
        self._minimal_update_frequency = minimal_update_frequency
        self._information_multiplier = information_multiplier

    def update_sparsity(self, model, step, acc_info):
        print(step, self._threshold, acc_info)
        time_since_last = step - self._step
        assert time_since_last >= 0
        if time_since_last < self._minimal_update_frequency:
            return
        self._step = step
        self._threshold -= self._threshold_decrease_per_iter * time_since_last
        self._threshold = max(self._threshold, self._minimal_threshold)
        self._threshold = min(self._threshold, self._maximal_threshold)
        print(f"Accuracy: {acc_info['acc']:.2%}; Threshold: {self._threshold:.2%}")
        if acc_info["acc"] > self._threshold:
            print(f"Originally using information (1 - sparsity) = {1 - model.sparsity_value:.10%}")
            model.sparsity_value = 1 - (1 - model.sparsity_value) * self._information_multiplier
            print(f"Now        using information (1 - sparsity) = {1 - model.sparsity_value:.10%}")
        self._threshold = max(self._threshold, acc_info["acc"])


def suo_types():
    return dict(NoopSUO=NoopSUO, LinearThresholdAdaptiveSUO=LinearThresholdAdaptiveSUO)
