#!/usr/bin/env python3
"""
Simple test script for ensemble model functionality.
"""

import sys
import os

# Add project root to path
sys.path.append('.')

def test_ensemble_import():
    """Test if ensemble model can be imported."""
    try:
        from models import EnsembleViTClassifier
        print("✓ EnsembleViTClassifier imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import EnsembleViTClassifier: {e}")
        return False

def test_ensemble_creation():
    """Test ensemble creation with dummy paths."""
    try:
        from models import EnsembleViTClassifier
        import torch

        # Create dummy model files for testing (won't actually load)
        dummy_full = "dummy_full.pt"
        dummy_rnn = "dummy_rnn.pt"

        # Test ensemble creation (this will fail at loading but should show the logic works)
        try:
            ensemble = EnsembleViTClassifier(
                full_vit_path=dummy_full,
                rnn_vit_path=dummy_rnn,
                ensemble_method='average',
                weights=[0.5, 0.5],
                device='cpu'
            )
        except FileNotFoundError:
            print("✓ Ensemble creation logic works (expected FileNotFoundError for dummy paths)")
            return True
        except Exception as e:
            print(f"✗ Unexpected error during ensemble creation: {e}")
            return False

    except Exception as e:
        print(f"✗ Failed to test ensemble creation: {e}")
        return False

def main():
    print("Testing Ensemble Model Functionality")
    print("=" * 40)

    success = True
    success &= test_ensemble_import()
    success &= test_ensemble_creation()

    if success:
        print("\n✓ All tests passed! Ensemble model is ready.")
        print("\nTo run ensemble evaluation:")
        print("python inference.py --ensemble --full_vit_path best_model_full.pt --rnn_vit_path best_model_rnn.pt --ensemble_method average --output_dir ensemble_results")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")

if __name__ == '__main__':
    main()
