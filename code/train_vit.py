import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append('.')

from dataset import PreprocessedDataset
from models import ViTClassifier
from utils import (
    MetricsCalculator,
    compute_class_weights,
    Trainer,
    WeightedBCELoss
)
from utils.config_utils import load_config, load_env, get_device, print_config, validate_config
from utils.data_analysis import analyze_dataset

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ViT model for ICH classification')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment/run')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    SEED = 5252
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS fallback enabled:", os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'))

    # Load environment variables and config
    env_vars = load_env()
    config = load_config(args.config)
    validate_config(config)

    print("\nConfiguration:")
    print_config(config)

    # Override experiment name if provided
    if args.experiment_name:
        env_vars['NOTEBOOK_NAME'] = args.experiment_name

    # Get device
    device = get_device(config['hardware']['device'])
    print(f"\nUsing device: {device}")

    # Create datasets
    data_root = config['data']['data_root']

    train_dataset = PreprocessedDataset(root_dir=os.path.join(data_root, 'train'))
    val_dataset = PreprocessedDataset(root_dir=os.path.join(data_root, 'val'))
    test_dataset = PreprocessedDataset(root_dir=os.path.join(data_root, 'test'))

    print("Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    print("\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Compute class weights from training data
    class_weights = compute_class_weights(
        train_dataset,
        strategy=config['loss']['pos_weight_strategy']
    )

    print("\nComputed class weights:")
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    for name, weight in zip(class_names, class_weights):
        print(f"  {name:20s}: {weight:.4f}")

    # Move to device
    class_weights = class_weights.to(device)

    # Create model - using ViTClassifier with sequence processing (like RNN but ViT-only)
    model = ViTClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        input_channels=config['model']['input_channels'],
        slice_channels=config['model'].get('slice_channels', 3),
        dropout=config['model']['dropout'],
        backend=config['model']['backend'],
        unfreeze_layers=config['model'].get('unfreeze_layers', 0)
    )

    model.freeze_backbone()

    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: {config['model']['name']}")
    print(f"Backend: {config['model']['backend']}")
    if hasattr(model, 'process_sequences') and model.process_sequences:
        print(f"Sequence processing: {model.num_slices} slices per sample")
    unfreeze_val = config['model'].get('unfreeze_layers', 0)
    if unfreeze_val == -1:
        print(f"Unfreeze layers: ALL (full fine-tuning)")
    elif unfreeze_val == 0:
        print(f"Unfreeze layers: NONE (frozen backbone)")
    else:
        print(f"Unfreeze layers: {unfreeze_val} (last N layers)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = WeightedBCELoss(pos_weights=class_weights)
    print("Using Weighted Binary Cross Entropy Loss")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
        print("Using Cosine Annealing LR Scheduler")
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
        print("Using Step LR Scheduler")
    else:
        scheduler = None
        print("No LR Scheduler")

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")

    # Initialize W&B
    USE_WANDB = not args.no_wandb

    if USE_WANDB:
        # Login to W&B (if not already logged in)
        if env_vars.get('WANDB_API_KEY'):
            wandb.login(key=env_vars['WANDB_API_KEY'])

        # Initialize run
        run = wandb.init(
            project=config['wandb']['project'],
            entity=env_vars.get('WANDB_ENTITY'),
            name=env_vars.get('NOTEBOOK_NAME', 'vit_ich_training'),
            config=config,
            tags=['vit', 'ich', 'multi-label', 'transformer']
        )

        # Watch model (log gradients and parameters)
        if config['wandb']['watch_model']:
            wandb.watch(model, log='all', log_freq=100)

        print(f"W&B initialized: {wandb.run.name}")
        print(f"  Project: {config['wandb']['project']}")
        print(f"  Run URL: {wandb.run.get_url()}")
    else:
        print("W&B disabled")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=USE_WANDB
    )

    # Train the model
    trainer.fit(epochs=config['training']['epochs'])

    # Load best checkpoint
    best_checkpoint_path = Path(config['checkpoint']['save_dir']) / 'best_model.pt'

    if best_checkpoint_path.exists():
        print(f"\nLoading best model from {best_checkpoint_path}")
        checkpoint = trainer.load_checkpoint(best_checkpoint_path)
        print(f"  Best validation {trainer.best_metric_name}: {checkpoint['best_metric']:.4f}")
    else:
        print("\nNo checkpoint found, using current model")

    # Evaluate on test set
    test_metrics = trainer.test(test_loader)

    # Create AUC-ROC plot with 7 curves
    class_names_plot = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural', 'Healthy', 'Multiple']
    class_names_key = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy', 'multiple']

    # Extract AUC values for all 7 classes
    aucs = []
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

    for name in class_names_key:
        auc_key = f'auc_roc_{name}'
        auc_value = test_metrics.get(auc_key, 0)
        aucs.append(auc_value)

    # Create bar plot for AUC-ROC values
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(class_names_plot, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Hemorrhage Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC-ROC Score', fontsize=14, fontweight='bold')
    ax.set_title('AUC-ROC Performance Across All Categories', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Customize x-axis labels
    ax.set_xticklabels(class_names_plot, rotation=45, ha='right', fontsize=12)

    # Add a legend explaining the categories
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#3498db', alpha=0.8, label='Individual Hemorrhage Types'),
        plt.Rectangle((0,0),1,1, facecolor='#1abc9c', alpha=0.8, label='Healthy Cases (No Hemorrhages)'),
        plt.Rectangle((0,0),1,1, facecolor='#e67e22', alpha=0.8, label='Multiple Hemorrhages')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint']['save_dir'], 'auc_roc_7curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Log to W&B
    if USE_WANDB:
        wandb.log({"auc_roc_7curves": wandb.Image(fig)})

        # Save test metrics to JSON
        results = {
            'model': config['model']['name'],
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in test_metrics.items()},
            'config': config
        }

        results_path = Path(config['checkpoint']['save_dir']) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")

        # Print summary
        print("\nFinal Results Summary")
        print("-" * 21)
        print(f"\nOverall Performance:")
        print(f"  AUC-ROC (Macro):      {test_metrics['auc_roc_macro']:.4f}")
        print(f"  AUC-ROC (Weighted):   {test_metrics.get('auc_roc_weighted', 0):.4f}")
        print(f"  F1-Score (Macro):     {test_metrics['f1_macro']:.4f}")
        print(f"  Exact Match Accuracy: {test_metrics['accuracy_exact']:.4f}")
        print(f"  Hamming Accuracy:     {test_metrics['accuracy_hamming']:.4f}")

        # Finish W&B run
        if USE_WANDB:
            wandb.finish()
            print("W&B run finished")


if __name__ == '__main__':
    main()
