from abc import ABC, abstractmethod


class SparsityUpdateOptimizer(ABC):
    """
    Not a torch optimizer object, but it does support the basic zero_grad/step interface.
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
    def update_sparsity(self, model, step, acc_info):
        """
        Does nothing
        """


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
        """
        The Linear Threshold Adaptive SUO uses an algorithm where a current accuracy threshold
            is mantained, and slowly decreased over time. If the model exceeds the threshold,
            the threshold is increased to match the model and the model's information is reduced
            (information + sparsity = 1).

        Arguments:
            optimizer: the underlying optimizer
            initial_threshold: the initial accuracy threshold [0-1]
            minimal_threshold: the minimal accuracy threshold to use [0-1]
            threshold_decrease_per_iter: the amount you decrease the threshold every iteration
            minimal_update_frequency: the minimal number of iterations between updates
            information_multiplier: how much to change the amount of information (1 - sparsity)
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
