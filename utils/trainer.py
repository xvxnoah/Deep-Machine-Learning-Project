import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import numpy as np

from .metrics import MetricsCalculator, log_classification_report
from .early_stopping import EarlyStopping
from .losses import mixup_criterion


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        config=None,
        use_wandb=True,
        mixup_fn=None,
        cutmix_fn=None
    ):
        """
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device (str): Device to use
            config (dict): Configuration dictionary
            use_wandb (bool): Whether to use W&B logging
            mixup_fn: Mixup augmentation function (optional)
            cutmix_fn: CutMix augmentation function (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.use_wandb = use_wandb
        self.mixup_fn = mixup_fn
        self.cutmix_fn = cutmix_fn
        
        # Gradient clipping
        self.grad_clip = config.get('training', {}).get('gradient_clip', None)
        # Gradient accumulation
        self.accumulation_steps = max(1, int(config.get('training', {}).get('accumulation_steps', 1)))
        
        # Scheduled backbone unfreezing
        unfreeze_cfg = config.get('training', {}).get('unfreeze', {})
        self.unfreeze_enabled = bool(unfreeze_cfg.get('enabled', False))
        self.unfreeze_epoch = int(unfreeze_cfg.get('start_epoch', 0))
        
        # Freeze backbone initially (will unfreeze later if enabled in config)
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
        
        # Mixup/CutMix probability
        self.mixup_cutmix_prob = config.get('augmentation', {}).get('mixup_cutmix_prob', 0.5)
        
        # Metrics
        self.num_classes = config.get('model', {}).get('num_classes', 5)
        self.train_metrics = MetricsCalculator(num_classes=self.num_classes)
        self.val_metrics = MetricsCalculator(num_classes=self.num_classes)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_metric = -float('inf')
        self.best_metric_name = config.get('checkpoint', {}).get('metric_for_best', 'val_auc_roc_macro')
        self.save_mode = config.get('checkpoint', {}).get('mode', 'max')
        
        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        if early_stop_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 7),
                min_delta=early_stop_config.get('min_delta', 0.0),
                mode=self.save_mode,
                verbose=True,
                restore_best_weights=early_stop_config.get('restore_best_weights', True)
            )
        else:
            self.early_stopping = None
        
        # Track training state
        self.current_epoch = 0
        self.global_step = 0
        # Thresholds
        self.current_thresholds = None
        self.best_thresholds = None
        
    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply mixup or cutmix augmentation
            use_mixup = False
            if self.mixup_fn is not None:
                # MixupCutmix returns either (images, labels) or (images, (labels_a, labels_b, lam))
                result = self.mixup_fn(images, labels)
                if isinstance(result, tuple) and len(result) == 2:
                    images, mixed_labels = result
                    # Check if labels were actually mixed
                    if isinstance(mixed_labels, tuple) and len(mixed_labels) == 3:
                        labels_a, labels_b, lam = mixed_labels
                        use_mixup = True
                    else:
                        # No mixing applied, use original labels
                        labels = mixed_labels
            
            # Forward pass
            if (batch_idx % self.accumulation_steps) == 0:
                self.optimizer.zero_grad()
            
            logits = self.model(images)
            
            # Calculate loss
            if use_mixup:
                loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
            else:
                loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Optimizer step every accumulation_steps
            if ((batch_idx + 1) % self.accumulation_steps) == 0:
                self.optimizer.step()
            
            # Update metrics (only with original labels, not mixed)
            total_loss += loss.item()
            if not use_mixup:
                self.train_metrics.update(logits, labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and batch_idx % self.config.get('wandb', {}).get('log_interval', 10) == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        # Final optimizer step if batches not divisible by accumulation_steps
        if (len(self.train_loader) % self.accumulation_steps) != 0:
            self.optimizer.step()

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        # TTA configuration
        eval_cfg = self.config.get('evaluation', {})
        tta_cfg = eval_cfg.get('tta', {})
        use_tta = bool(tta_cfg.get('enabled', False))
        tta_hflip = bool(tta_cfg.get('hflip', True))
        tta_vflip = bool(tta_cfg.get('vflip', False))

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            if use_tta:
                logits_accum = logits
                num_aug = 1
                if tta_hflip:
                    logits_h = self.model(torch.flip(images, dims=[-1]))
                    logits_accum = logits_accum + logits_h
                    num_aug += 1
                if tta_vflip:
                    logits_v = self.model(torch.flip(images, dims=[-2]))
                    logits_accum = logits_accum + logits_v
                    num_aug += 1
                logits = logits_accum / float(num_aug)
            loss = self.criterion(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            self.val_metrics.update(logits, labels)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        # Optimize per-class thresholds on validation set
        try:
            self.current_thresholds = self.val_metrics.optimize_thresholds(
                method='f1', per_class=True
            )
            # Compute metrics with optimized thresholds for reporting
            metrics_opt = self.val_metrics.compute_with_thresholds(self.current_thresholds)
        except Exception:
            # Fallback if optimization fails
            self.current_thresholds = None
            metrics_opt = {}

        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss
        # Add optional optimized F1 for visibility
        if 'f1_macro' in metrics_opt:
            metrics['f1_macro_opt'] = metrics_opt['f1_macro']
        
        return metrics
    
    def fit(self, epochs):
        print(f"\n{'='*80}")
        print(f"Starting Training for {epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            # Unfreeze schedule
            if self.unfreeze_enabled and epoch == self.unfreeze_epoch:
                try:
                    self.model.unfreeze_backbone()
                    print(f"Backbone unfrozen at epoch {epoch}")
                except Exception:
                    pass
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            if self.use_wandb:
                log_dict = {}
                for k, v in train_metrics.items():
                    log_dict[f'train/{k}'] = v
                for k, v in val_metrics.items():
                    log_dict[f'val/{k}'] = v
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
            
            # Print metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train AUC: {train_metrics.get('auc_roc_macro', 0):.4f} | "
                  f"Train F1: {train_metrics.get('f1_macro', 0):.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | "
                  f"Val AUC:   {val_metrics.get('auc_roc_macro', 0):.4f} | "
                  f"Val F1:   {val_metrics.get('f1_macro', 0):.4f}")
            
            # Save checkpoint
            current_metric = val_metrics.get(self.best_metric_name.replace('val_', ''), 0)
            is_best = False
            
            if self.save_mode == 'max':
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            
            # Save checkpoint (only best model)
            if is_best:
                # Persist thresholds from current validation
                self.best_thresholds = self.current_thresholds
                self.save_checkpoint(epoch, is_best, val_metrics)
            
            # Check early stopping
            if self.early_stopping is not None:
                if self.early_stopping(current_metric, self.model, epoch):
                    print(f"\nStopping early at epoch {epoch}")
                    break
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best {self.best_metric_name}: {self.best_metric:.4f}")
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, epoch, is_best=False, metrics=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config,
            'thresholds': self.best_thresholds
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save best model only
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"  ✓ Best model saved: {best_path}")
        
        # Log to W&B
        if self.use_wandb and self.config.get('wandb', {}).get('log_model', True):
            wandb.save(str(best_path))
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        self.best_thresholds = checkpoint.get('thresholds', None)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint
    
    @torch.no_grad()
    def test(self, test_loader):
        print(f"\n{'='*80}")
        print("Running Test Evaluation")
        print(f"{'='*80}\n")
        
        self.model.eval()
        test_metrics = MetricsCalculator(num_classes=self.num_classes)
        test_metrics.reset()
        
        total_loss = 0
        progress_bar = tqdm(test_loader, desc='Testing')
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            test_metrics.update(logits, labels)
        
        # Apply stored best thresholds if available
        if self.best_thresholds is not None:
            try:
                test_metrics.set_thresholds(self.best_thresholds)
                print("Using optimized per-class thresholds for test evaluation.")
            except Exception:
                pass

        # Compute metrics
        avg_loss = total_loss / len(test_loader)
        metrics = test_metrics.compute()
        metrics['loss'] = avg_loss
        
        # Log to W&B
        if self.use_wandb:
            log_dict = {f'test/{k}': v for k, v in metrics.items()}
            wandb.log(log_dict)
        
        # Print report
        log_classification_report(metrics, prefix='test')
        
        return metrics

