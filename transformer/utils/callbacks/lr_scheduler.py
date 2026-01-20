"""
Learning Rate Scheduler for Transformer.

Implements the warmup + inverse square root decay schedule
from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLRScheduler(_LRScheduler):
    """
    Transformer Learning Rate Scheduler.

    Implements the learning rate schedule from the original Transformer paper:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This increases linearly during warmup, then decreases proportionally
    to the inverse square root of the step number.

    Args:
        optimizer: Wrapped optimizer
        d_model: Model dimension (used for scaling)
        warmup_steps: Number of warmup steps
        factor: Scaling factor for the learning rate
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step_count = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        self._step_count += 1
        step = self._step_count

        # Calculate learning rate
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))

        lr = self.factor * (self.d_model ** (-0.5)) * min(arg1, arg2)

        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        # Override step to avoid epoch-based scheduling
        if epoch is None:
            self._step_count += 1
            values = self.get_lr()
            self._step_count -= 1  # Will be incremented in get_lr

            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr

            self._last_lr = values
        else:
            super().step(epoch)


class WarmupScheduler(_LRScheduler):
    """
    Simple Warmup Learning Rate Scheduler.

    Linearly increases learning rate during warmup, then keeps it constant
    or applies another scheduler.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        target_lr: Target learning rate after warmup
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        target_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self._step_count = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        self._step_count += 1
        step = self._step_count

        if step <= self.warmup_steps:
            # Linear warmup
            lr = self.target_lr * (step / self.warmup_steps)
        else:
            lr = self.target_lr

        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            self._step_count += 1
            values = self.get_lr()
            self._step_count -= 1

            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr

            self._last_lr = values
        else:
            super().step(epoch)


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine Annealing with Warmup.

    Linearly increases learning rate during warmup, then applies
    cosine annealing to the target learning rate.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        target_lr: Target (peak) learning rate
        min_lr: Minimum learning rate at the end
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        target_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.min_lr = min_lr
        self._step_count = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        self._step_count += 1
        step = self._step_count

        if step <= self.warmup_steps:
            # Linear warmup
            lr = self.target_lr * (step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            self._step_count += 1
            values = self.get_lr()
            self._step_count -= 1

            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr

            self._last_lr = values
        else:
            super().step(epoch)
