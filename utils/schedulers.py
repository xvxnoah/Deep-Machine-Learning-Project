"""
Learning Rate Schedulers

This module implements Cosine Annealing with Warm Restarts scheduler.
The scheduler periodically restarts the learning rate to escape local minima.
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts scheduler.
    
    Restarts the learning rate periodically to escape local minima.
    After each restart, the period can be multiplied by T_mult.
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of epochs for the first restart
        T_mult: Factor to multiply T_i after each restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler that gradually increases learning rate.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        after_scheduler: Scheduler to use after warmup
        warmup_lr_init: Initial learning rate for warmup (default: 0)
    """
    
    def __init__(self, optimizer, warmup_epochs, after_scheduler, warmup_lr_init=0):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_lr_init = warmup_lr_init
        self.finished_warmup = False
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr_init + (base_lr - self.warmup_lr_init) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                self.after_scheduler.base_lrs = self.base_lrs
            return self.after_scheduler.get_lr()
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                self.after_scheduler.base_lrs = self.base_lrs
            self.after_scheduler.step(epoch - self.warmup_epochs if epoch is not None else None)
            self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epochs


def create_scheduler(optimizer, config, steps_per_epoch=None):
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        steps_per_epoch: Number of training steps per epoch (for step-based schedulers)
    
    Returns:
        Learning rate scheduler
    """
    training_config = config['training']
    scheduler_type = training_config['scheduler']
    
    # Get common parameters
    warmup_epochs = int(training_config.get('warmup_epochs', 0))
    warmup_lr_init = float(training_config.get('warmup_lr_init', 0))
    
    # Create the main scheduler
    if scheduler_type == 'cosine_warm_restarts':
        min_lr = float(training_config.get('min_lr', 1e-7))
        T_0 = int(training_config.get('restart_period', 15))
        T_mult = int(training_config.get('restart_mult', 2))
        
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Wrap with warmup if specified
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            after_scheduler=main_scheduler,
            warmup_lr_init=warmup_lr_init
        )
    else:
        scheduler = main_scheduler
    
    return scheduler
