#!/usr/bin/env python3
"""
Create Architecture Diagram for ViT-based Brain Hemorrhage Classification
Simple text-based diagram since matplotlib has issues in this environment
"""

def create_text_diagram():
    """Create a comprehensive text-based architecture diagram."""

    diagram = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                BRAIN HEMORRHAGE CLASSIFICATION ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────────────────┤

                    Input CT Scan (9-channel)
                    3 slices × 3 window presets
                                │
                                ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Channel Adapter                                      │
│                    9 → 3 channels using:                                   │
│                 • Conv1x1 (learnable 1×1 convolution)                       │
│                 • Linear Projection (per spatial location)                  │
│                 • Average Pooling (group channels)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │                 │
                    │                 │
          ┌─────────▼─────────┐       ▼─────────┐
          │                   │                 │
          │   ViTClassifier   │   ViT-B/16      │
          │  (Plain ViT Model)│   Encoder       │
          │                   │   +             │
          │                   │                 │
          │                   │   Bi-directional │
          │                   │   LSTM          │
          │                   │   (512 hidden)  │
          │                   │                 │
          └─────────┬─────────┘   └─────────┬───┘
                    │                       │
                    │                       │
                    ▼                       ▼
          ┌─────────────────────────────────────────┐
          │              Pooling &                  │
          │        Classification Head              │
          │      • LayerNorm + Dropout              │
          │      • Linear Layer (→ 5 classes)       │
          │      • Sigmoid Activation               │
          └─────────────────────────────────────────┘
                    │                       │
                    │                       │
                    ▼                       ▼

          ┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
          │         ViT Model Output                │    │        ViT+RNN Model Output             │
          │      Sigmoid (5 classes)               │    │     Sigmoid (5 classes)                │
          └─────────────────────────────────────────┘    └─────────────────────────────────────────┘
                    │                       │
                    │                       │
                    └─────────┬─────────────┘
                              │
                              ▼

          ┌─────────────────────────────────────────┐
          │         Ensemble Combination           │
          │    • Average (equal weights)           │
          │    • Weighted (configurable)           │
          │    • Max Confidence (best model)       │
          │    • Voting (majority prediction)      │
          └─────────────────────────────────────────┘
                              │
                              ▼

          ┌─────────────────────────────────────────┐
          │          Final Predictions             │
          │   5 Hemorrhage Types:                  │
          │   • Epidural                           │
          │   • Intraparenchymal                   │
          │   • Intraventricular                   │
          │   • Subarachnoid                       │
          │   • Subdural                           │
          │                                        │
          │   + Healthy Cases (no hemorrhages)     │
          │   + Multiple Cases (>1 hemorrhage)     │
          └─────────────────────────────────────────┘

Implementation Details:
• ViT Backend: HuggingFace Transformers (google/vit-base-patch16-224)
• Input Resolution: 224×224 pixels
• Training: Multi-label classification with weighted BCE loss
• Optimization: AdamW with cosine annealing scheduler
• Evaluation: AUC-ROC, F1-Score, Hamming/Exact Match Accuracy

    """

    return diagram

if __name__ == "__main__":
    diagram = create_text_diagram()

    # Save as text file
    with open('architecture_diagram.txt', 'w') as f:
        f.write(diagram)

    print(diagram)
    print("\n✓ Architecture diagram saved as 'architecture_diagram.txt'")
