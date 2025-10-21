#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -o ./slurm-output/%x-%A.out
#SBATCH -e ./slurm-output/%x-%A.err

# SLURM script for ViT training (no RNN, HuggingFace ViT with 9→3 channel adapter)

echo "Job is running on $(hostname)"
echo "Num cores: $(nproc)"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"

# Print GPU information
nvidia-smi

# Print current directory and list files
pwd
ls -la

# Activate the virtual environment
echo "Activating virtual environment..."
source a5venv/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully"
    which python3
    python3 --version
else
    echo "Failed to activate virtual environment"
    exit 1
fi

# Run the ViT training script (no RNN, HuggingFace ViT with channel adapter)
CONFIG_FILE=${1:-"configs/base_config.yaml"}
echo "Starting ViT training with HuggingFace ViT and channel adapter with config: $CONFIG_FILE"
echo "Using WeightedBCE loss, AdamW optimizer, cosine scheduler"
echo "Early stopping monitors macro AUC (target: 0.8886)"
echo "Channel adapter handles 9→3 channels, includes AUC for healthy/multiple cases"
python3 train_vit.py --config $CONFIG_FILE

echo "Training completed!"
