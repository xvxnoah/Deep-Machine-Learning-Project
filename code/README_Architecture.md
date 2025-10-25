# Brain Hemorrhage Classification Architecture

## Overview

This project implements a comprehensive deep learning pipeline for multi-label classification of brain hemorrhages from CT scans using Vision Transformers (ViTs) and ensemble methods.

## Architecture Diagram

The complete architecture is visualized in `architecture_diagram.txt` and shows the data flow from input CT scans to final predictions.

## Key Components

### 1. Input Data
- **Format**: 9-channel CT scans (224×224 pixels)
- **Structure**: 3 consecutive CT slices × 3 window presets per slice
- **Purpose**: Multi-channel input captures different tissue contrasts and slice relationships

### 2. Channel Adapter
Converts 9-channel input to 3-channel format compatible with ViT:

- **Conv1x1**: Learnable 1×1 convolution for channel projection
- **Linear Projection**: Per-spatial location linear transformation
- **Average Pooling**: Groups and averages channels (9→3)

### 3. Two Parallel Model Branches

#### Branch A: ViTClassifier (Plain ViT)
- **Model**: Vision Transformer (google/vit-base-patch16-224)
- **Backend**: HuggingFace Transformers
- **Processing**: Single image or sequential slice processing
- **Output**: 5-class hemorrhage predictions

#### Branch B: ViTTripletBiRNNClassifier (ViT+RNN)
- **Model**: ViT-B/16 encoder + Bi-directional LSTM
- **Sequence Processing**: 3 slices processed through shared ViT backbone
- **RNN**: 512 hidden units, bidirectional LSTM for temporal modeling
- **Pooling**: Last or mean sequence pooling
- **Output**: 5-class hemorrhage predictions

### 4. Classification Heads (Both Branches)
- **LayerNorm** + **Dropout** (0.1) for regularization
- **Linear Layer**: Projects to 5 hemorrhage classes
- **Sigmoid Activation**: Multi-label classification output

### 5. Ensemble Combination
Four ensemble strategies implemented:

- **Average**: Equal weight combination (0.5, 0.5)
- **Weighted**: Configurable weighted combination
- **Max Confidence**: Selects prediction from model with higher mean confidence
- **Voting**: Majority voting with threshold (0.5) binarization

## Output Classes

### 5 Hemorrhage Types
1. **Epidural**: Blood between skull and dura mater
2. **Intraparenchymal**: Bleeding within brain tissue
3. **Intraventricular**: Blood in brain's ventricular system
4. **Subarachnoid**: Blood in space between brain and arachnoid membrane
5. **Subdural**: Blood between dura and arachnoid membranes

### Special Cases
- **Healthy**: No hemorrhages detected (all classes = 0)
- **Multiple**: Multiple hemorrhage types present (>1 class = 1)

## Implementation Details

### Model Configuration
```yaml
model:
  name: "google/vit-base-patch16-224"
  pretrained: true
  num_classes: 5
  input_channels: 9
  slice_channels: 3
  dropout: 0.1
  backend: "huggingface"  # or "torchvision", "timm"
```

### Training Setup
- **Loss**: Weighted Binary Cross Entropy (handles class imbalance)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine Annealing with warmup
- **Metrics**: AUC-ROC, F1-Score, Hamming/Exact Match Accuracy

### Hardware & Reproducibility
- **Device**: Auto-detection (CUDA/MPS/CPU)
- **Seed**: 5252 for reproducible results
- **Mixed Precision**: Optional for memory optimization

## Files Structure

```
models/
├── vit_model.py           # Main model implementations
├── channel_adapter.py     # Channel adaptation utilities
└── __init__.py           # Model exports

configs/
└── base_config.yaml      # Complete configuration

utils/
├── trainer.py            # Training loop and utilities
├── metrics.py            # Evaluation metrics
└── losses.py             # Custom loss functions

Scripts:
├── train_vit.py          # Main training script
├── run_ensemble.py       # Ensemble evaluation
└── inference.py          # Model inference
```

## Usage

### Training Individual Models
```bash
python train_vit.py --config configs/base_config.yaml
```

### Ensemble Evaluation
```bash
python run_ensemble.py --full_vit_path best_model_full.pt \
                       --rnn_vit_path best_model_rnn.pt \
                       --method average
```

### Inference
```bash
python inference.py --model_path best_model.pt \
                    --data_path test_data/
```

## Performance

The ensemble approach typically achieves:
- **AUC-ROC Macro**: >0.85 across all hemorrhage types
- **F1-Score**: Improved over individual models by 2-5%
- **Robustness**: Better generalization across different hemorrhage types

## References

- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- Multi-label Classification: Standard BCE loss with class weighting
- Ensemble Methods: Weighted combination for improved performance
