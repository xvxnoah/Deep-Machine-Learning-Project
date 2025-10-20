import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import numpy as np

from .metrics import MetricsCalculator, log_classification_report


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
        use_wandb=True
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
        
        # Gradient clipping
        self.grad_clip = config.get('training', {}).get('gradient_clip', None)
        
        # Metrics
        self.num_classes = config.get('model', {}).get('num_classes', 5)
        self.train_metrics = MetricsCalculator(num_classes=self.num_classes)
        self.val_metrics = MetricsCalculator(num_classes=self.num_classes)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_metric_name = config.get('checkpoint', {}).get('metric_for_best', 'val_auc_roc_macro')
        self.save_mode = config.get('checkpoint', {}).get('mode', 'max')
        
        # Initialize best_metric according to save_mode so comparisons make sense
        if self.save_mode == 'min':
            self.best_metric = float('inf')
        else:
            self.best_metric = -float('inf')
        
        # Early stopping config (defaults under training.early_stopping)
        es_cfg = config.get('training', {}).get('early_stopping', {})
        self.early_stopping = bool(es_cfg.get('enabled', False))
        self.early_stopping_patience = int(es_cfg.get('patience', 10))
        self.early_stopping_min_delta = float(es_cfg.get('min_delta', 0.0))
        # mode can override save_mode, but default to save_mode
        self.early_stopping_mode = es_cfg.get('mode', self.save_mode)
        self.no_improve_epochs = 0
        
        # Track training state
        self.current_epoch = 0
        self.global_step = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
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
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            self.val_metrics.update(logits, labels)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(self, epochs):
        print(f"\n{'='*80}")
        print(f"Starting Training for {epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
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
            current_metric = val_metrics.get(self.best_metric_name.replace('val_', ''), None)
            if current_metric is None:
                current_metric = 0

            # Determine whether this epoch improved the monitored metric
            is_best = False
            improved = False
            if self.early_stopping_mode == 'min':
                if current_metric < self.best_metric - self.early_stopping_min_delta:
                    improved = True
            else:
                if current_metric > self.best_metric + self.early_stopping_min_delta:
                    improved = True

            if improved:
                self.best_metric = current_metric
                is_best = True
                self.no_improve_epochs = 0
            else:
                # No improvement this epoch
                self.no_improve_epochs += 1

            # Save checkpoint (periodically or when new best)
            if epoch % self.config.get('checkpoint', {}).get('save_frequency', 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best, val_metrics)

            # Early stopping: stop if no improvement for configured patience
            if self.early_stopping and self.no_improve_epochs >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {self.no_improve_epochs} epochs).")
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
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved: {best_path}")
            
            # Log to W&B
            if self.use_wandb and self.config.get('wandb', {}).get('log_model', True):
                wandb.save(str(best_path))
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        
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

